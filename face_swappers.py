import torch
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
import onnx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.helpers.downloader import download_file
from app.helpers.miscellaneous import is_file_exists
class FaceSwappers:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        if not self.models_processor.models[arcface_model]:
            self.models_processor.models[arcface_model] = self.models_processor.load_model(arcface_model)

        if arcface_model == 'CSCSArcFace':
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(arcface_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image
        
    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.models_processor.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 112, 112)

        if arcface_model == 'Inswapper128ArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
        elif arcface_model == 'SimSwapArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = torch.div(img.to(torch.float32), 255.0)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
        else:
            cropped_image = img.permute(1,2,0).clone() #112,112,3
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            # Normalize
            img = torch.div(img, 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = self.models_processor.models[arcface_model].get_inputs()[0].name

        outputs = self.models_processor.models[arcface_model].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.models_processor.models[arcface_model].io_binding()
        io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[arcface_model].run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, self.models_processor.FFHQ_kps)

        temp = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)
        
        image = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(temp)
        
        cropped_image = image.permute(1, 2, 0).clone()
        if image.dtype == torch.uint8:
            image = torch.div(image.to(torch.float32), 255.0)

        image = v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

        # Ritorna l'immagine e l'immagine ritagliata
        return torch.unsqueeze(image, 0).contiguous(), cropped_image  # (C, H, W) e (H, W, C)

    def recognize_cscs(self, img, face_kps):
        # Usa la funzione di preprocessamento
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models_processor.models['CSCSArcFace'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device)

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

        self.models_processor.models['CSCSArcFace'].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding = torch.from_numpy(output).to('cpu')
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)
        embedding = embedding + embedding_id

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        if not self.models_processor.models['CSCSIDArcFace']:
            self.models_processor.models['CSCSIDArcFace'] = self.models_processor.load_model('CSCSIDArcFace')

        # Use preprocess_image_cscs when face_kps is not None. When it is None img is already preprocessed.
        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models_processor.models['CSCSIDArcFace'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device)

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
            
        self.models_processor.models['CSCSIDArcFace'].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding_id = torch.from_numpy(output).to('cpu')
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()

    def calc_swapper_latent_cscs(self, source_embedding):
        latent = source_embedding.reshape((1,-1))
        return latent

    def run_swapper_cscs(self, image, embedding, output):
        if not self.models_processor.models['CSCS']:
            self.models_processor.models['CSCS'] = self.models_processor.load_model('CSCS')

        io_binding = self.models_processor.models['CSCS'].io_binding()
        io_binding.bind_input(name='input_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='input_2', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['CSCS'].run_with_iobinding(io_binding)

    def calc_inswapper_latent(self, source_embedding):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_inswapper(self, image, embedding, output):
        if not self.models_processor.models['Inswapper128']:
            self.models_processor.models['Inswapper128'] = self.models_processor.load_model('Inswapper128')

        io_binding = self.models_processor.models['Inswapper128'].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Inswapper128'].run_with_iobinding(io_binding)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_iss_swapper(self, image, embedding, output, version="A"):
        ISS_MODEL_NAME = f'InStyleSwapper256 Version {version}'
        if not self.models_processor.models[ISS_MODEL_NAME]:
            self.models_processor.models[ISS_MODEL_NAME] = self.models_processor.load_model(ISS_MODEL_NAME)
        
        io_binding = self.models_processor.models[ISS_MODEL_NAME].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[ISS_MODEL_NAME].run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        #latent /= np.linalg.norm(latent)
        latent = latent/np.linalg.norm(latent,axis=1,keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.models_processor.models['SimSwap512']:
            self.models_processor.models['SimSwap512'] = self.models_processor.load_model('SimSwap512')

        io_binding = self.models_processor.models['SimSwap512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['SimSwap512'].run_with_iobinding(io_binding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        ghostfaceswap_model, output_name = None, None
        if swapper_model == 'GhostFace-v1':
            if not self.models_processor.models['GhostFacev1']:
                self.models_processor.models['GhostFacev1'] = self.models_processor.load_model('GhostFacev1')

            ghostfaceswap_model = self.models_processor.models['GhostFacev1']
            output_name = '781'

        elif swapper_model == 'GhostFace-v2':
            if not self.models_processor.models['GhostFacev2']:
                self.models_processor.models['GhostFacev2'] = self.models_processor.load_model('GhostFacev2')

            ghostfaceswap_model = self.models_processor.models['GhostFacev2']
            output_name = '1165'

        elif swapper_model == 'GhostFace-v3':
            if not self.models_processor.models['GhostFacev3']:
                self.models_processor.models['GhostFacev3'] = self.models_processor.load_model('GhostFacev3')

            ghostfaceswap_model = self.models_processor.models['GhostFacev3']
            output_name = '1549'

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name=output_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def swap_core_simplified(self, img, kps_5, s_e=None, t_e=None, parameters=None, processor=None):
        # A simplified version of swap_core for use in non-GUI environments
        from app.processors.utils import faceutil
        from math import floor, ceil
        from torchvision.transforms import v2
        import torchvision.transforms as transforms

        s_e = s_e if isinstance(s_e, np.ndarray) else []
        t_e = t_e if isinstance(t_e, np.ndarray) else []
        parameters = parameters or {}
        swapper_model = parameters.get('SwapModelSelection', 'Inswapper128')

        # Scaling transforms
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t384 = v2.Resize((384, 384), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Get tform
        tform = trans.SimilarityTransform()
        if swapper_model not in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'):
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M

        # Grab 512 face from image and create scaled versions
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        original_face_384 = t384(original_face_512)

        dim = 1
        latent = None
        input_face_affined = original_face_128
        
        # Swapper logic
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = self.calc_inswapper_latent(s_e)
            dim = 1
            input_face_affined = original_face_128
        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = self.calc_swapper_latent_iss(s_e, version)
            dim = 2
            input_face_affined = original_face_256
        elif swapper_model == 'SimSwap512':
            latent = self.calc_swapper_latent_simswap512(s_e)
            dim = 4
            input_face_affined = original_face_512
        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            latent = self.calc_swapper_latent_ghost(s_e)
            dim = 2
            input_face_affined = original_face_256
        elif swapper_model == 'CSCS':
            latent = self.calc_swapper_latent_cscs(s_e)
            dim = 2
            input_face_affined = original_face_256

        if latent is not None:
            latent_tensor = torch.from_numpy(latent).float().to(self.models_processor.device)
            output_size = 128 * dim
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined_norm = input_face_affined.permute(1, 2, 0) / 255.0
            
            # Simple run (itex=1)
            if swapper_model == 'Inswapper128':
                input_face_disc = input_face_affined_norm.permute(2, 0, 1).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.run_inswapper(input_face_disc, latent_tensor, swapper_output)
                swap = t512(swapper_output.squeeze(0) * 255.0)
            elif swapper_model.startswith('InStyleSwapper'):
                input_face_disc = input_face_affined_norm.permute(2, 0, 1).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.run_iss_swapper(input_face_disc, latent_tensor, swapper_output, swapper_model[-1])
                swap = t512(swapper_output.squeeze(0) * 255.0)
            elif swapper_model == 'SimSwap512':
                input_face_disc = input_face_affined_norm.permute(2, 0, 1).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.run_swapper_simswap512(input_face_disc, latent_tensor, swapper_output)
                swap = swapper_output.squeeze(0) * 255.0
            elif swapper_model.startswith('GhostFace'):
                input_face_disc = (input_face_affined_norm.permute(2, 0, 1) * 2.0 - 1.0).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.run_swapper_ghostface(input_face_disc, latent_tensor, swapper_output, swapper_model)
                swap = t512((swapper_output.squeeze(0) * 127.5 + 127.5))
            elif swapper_model == 'CSCS':
                input_face_disc = ((input_face_affined_norm.permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.run_swapper_cscs(input_face_disc, latent_tensor, swapper_output)
                swap = t512((swapper_output.squeeze(0) * 0.5 + 0.5) * 255.0)
            else:
                swap = original_face_512
        else:
            swap = original_face_512

        # Create border mask
        border_mask = torch.ones((1, 128, 128), dtype=torch.float32, device=self.models_processor.device)
        top = parameters.get('BorderTopSlider', 10)
        left = parameters.get('BorderLeftSlider', 10)
        right = 128 - parameters.get('BorderRightSlider', 10)
        bottom = 128 - parameters.get('BorderBottomSlider', 10)
        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0
        gauss = transforms.GaussianBlur(15, 3)
        border_mask = gauss(border_mask)
        swap_mask = t512(border_mask)
        
        swap = torch.mul(swap, swap_mask)

        # Paste back
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])
        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
        l, t, r, b = floor(np.min(x)), floor(np.min(y)), ceil(np.max(x)), ceil(np.max(y))
        l, t = max(0, l), max(0, t)
        r, b = min(img.shape[2], r), min(img.shape[1], b)

        swap_full = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_full = v2.functional.affine(swap_full, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        
        swap_mask_full = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask_full = v2.functional.affine(swap_mask_full, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        
        img_crop = img[:, t:b, l:r]
        swap_crop = swap_full[:, t:b, l:r]
        mask_crop = swap_mask_full[:, t:b, l:r]
        
        img[:, t:b, l:r] = (swap_crop * mask_crop + img_crop * (1 - mask_crop)).type(torch.uint8)
        
        return img, None, None