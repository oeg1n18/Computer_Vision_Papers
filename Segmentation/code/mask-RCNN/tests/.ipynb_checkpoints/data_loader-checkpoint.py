import torch

class TestDataLoader:
    def __init__(self, dataloader, test_batches=5):
        self.dl = dataloader
        self.test_batches = test_batches
        
    def test_img_color_channels(self):
        pass_flag = True
        
        for i, (images, targs) in enumerate(self.dl):
            for image in images:
                if not image.shape[0] == 3:
                    pass_flag = False
            if i > self.test_batches:
                break
        return pass_flag
    
    def test_img_range(self):
        pass_flag = True
        
        for i, (images, targs) in enumerate(self.dl):
            for image in images:
                if image.min() < 0 or image.max() > 1:
                    pass_flag = False
            if i > self.test_batches:
                    break
            
        return pass_flag
    
    def test_img_dtype(self):
        pass_flag = True
        
        for i, (images, targs) in enumerate(self.dl):
            for image in images:
                if image.dtype != torch.float32:
                    pass_flag = False
            if i > self.test_batches:
                break
        return pass_flag
    
    def targ_boxes_shape(self):
        pass_flag = True
        for i, (images, targs) in enumerate(self.dl):
            for targ in targs:
                if targ["boxes"].shape[1] != 4:
                    pass_flag=False
            if i > self.test_batches:
                break
        return pass_flag
    
    def targ_boxes_fit_in_image(self):
        pass_flag = True
        for i, (images, targs) in enumerate(self.dl):
            for i in range(len(images)):
                img = images[i]
                targ = targs[i]
                
                for j in range(targ["boxes"].shape[0]):
                    box = targ["boxes"][j, :]
                
                    maxx = max(box[0], box[2])
                    maxy = max(box[1], box[3])

                    if maxx > img.shape[2] or maxy > img.shape[1]:
                        print(maxx, maxy, img.shape)
                        pass_flag=False
                        break      
            if i > self.test_batches:
                break
        return pass_flag
    
    def targ_boxes_shape(self):
        pass_flag = True
        
        for i, (images, targs) in enumerate(self.dl):
            for targ in targs:
                for j in range(targ["boxes"].shape[0]):
                    if targ["boxes"][j, 0] >= targ["boxes"][j, 2]:
                        pass_flag=False
                    if targ["boxes"][j, 1] >= targ["boxes"][j,3]:
                        pass_flag=False
            if i > self.test_batches:
                break
        return pass_flag
    
    def mask_shape(self):
        pass_flag = True
        
        for i, (images, targs) in enumerate(self.dl):
            for i in range(len(images)):
                img = images[i]
                targ = targs[i]
                
                for mask in targ["masks"]:
                    if img.shape[1] != mask.shape[0] or img.shape[2] != mask.shape[1]:
                        pass_flag = False
                    if i > self.test_batches:
                        break
        return pass_flag
    
        
    def mask_type(self):
        pass_flag = True
        
        for i, (image, targs) in enumerate(self.dl):
            for targ in targs:
                for mask in targ["masks"]:
                    if mask.dtype != torch.float32:
                        pass_flag = False
            if i > self.test_batches:
                break
        return pass_flag 
    

    def process_tests(self):
        print("Test 1: ", self.test_img_color_channels())
        print("Test 2: ", self.test_img_range())
        print("Test 3: ", self.test_img_dtype())
        print("Test 4: ", self.targ_boxes_shape())
        print("Test 5: ", self.targ_boxes_fit_in_image())
        print("Test 6: ", self.targ_boxes_shape())
        print("Test 7: ", self.mask_shape())
        print("Test 8: ", self.mask_type())
        