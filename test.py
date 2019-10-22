# encoding: utf-8

from __future__ import print_function
import argparse, os, pickle, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from predictor import ConditionalMotionNet, ConditionalAppearanceNet
from encoder import define_E
from util import generateLoop, videoWrite, normalize, denormalize

class AnimatingLandscape():
    
    def __init__(self, args):
        self.model_path = args.model_path
        self.model_epoch = args.model_epoch
        self.gpu = int(args.gpu)
        self.input_path = args.input
        self.outdir_path = args.outdir
        self.t_m = float(args.motion_latent_code)
        self.s_m = float(args.motion_speed)
        self.t_a = float(args.appearance_latent_code)
        self.s_a = float(args.appearance_speed)
        self.t_m = min(1.,max(0.,self.t_m))
        self.s_m = min(1.,max(1e-3,self.s_m))
        self.t_a = min(1.,max(0.,self.t_a))
        self.s_a = min(1.,max(1e-3,self.s_a))
        self.TM = int(args.motion_frame_number)
        self.w, self.h = 256, 256 #Image size for network input
        self.fw, self.fh = None, None #Output image size
        self.pad = 64 #Reflection padding size for sampling outside of the image
    
    def PredictMotion(self):  
        print('Motion: ')
        P_m = ConditionalMotionNet()
        param = torch.load(self.model_path + '/PMNet_weight_' + self.model_epoch + '.pth')
        P_m.load_state_dict(param)
        if self.gpu>-1:
            P_m.cuda(self.gpu)
        
        with open(self.model_path + '/codebook_m_' + self.model_epoch + '.pkl', 'rb') as f:
            codebook_m = pickle.load(f) if sys.version_info[0] == 2 else pickle.load(f, encoding='latin1')
    
        id1 = int(np.floor((len(codebook_m)-1)*self.t_m))
        id2 = int(np.ceil((len(codebook_m)-1)*self.t_m))
        z_weight = (len(codebook_m)-1)*self.t_m-np.floor((len(codebook_m)-1)*self.t_m)
        z_m = (1.-z_weight)*codebook_m[id1:id1+1]+z_weight*codebook_m[id2:id2+1]
        z_m = Variable(torch.from_numpy(z_m.astype(np.float32)))
        if self.gpu>-1:
            z_m = z_m.cuda(self.gpu)
        initial_coordinate = np.array([np.meshgrid(np.linspace(-1,1,self.w+2*self.pad), np.linspace(-1,1,self.h+2*self.pad), sparse=False)]).astype(np.float32)
        initial_coordinate = Variable(torch.from_numpy(initial_coordinate))
        if self.gpu>-1:
            initial_coordinate = initial_coordinate.cuda(self.gpu)
        
        with torch.no_grad():
            
            test_img = cv2.imread(self.input_path)
            test_img = cv2.resize(test_img, (self.w,self.h))
            test_input = np.array([normalize(test_img)])
            test_input = Variable(torch.from_numpy(test_input.transpose(0,3,1,2)))       
            if self.gpu>-1:
                test_input = test_input.cuda(self.gpu)
            padded_test_input = F.pad(test_input, (self.pad,self.pad,self.pad,self.pad), mode='reflect')
            
            test_img_large = cv2.imread(self.input_path)
            if self.fw == None or self.fh == None:
                self.fh, self.fw = test_img_large.shape[:2]
            test_img_large = cv2.resize(test_img_large, (self.fw, self.fh))
            padded_test_input_large = np.array([normalize(test_img_large)])
            padded_test_input_large = Variable(torch.from_numpy(padded_test_input_large.transpose(0,3,1,2)))
            if self.gpu>-1:
                padded_test_input_large = padded_test_input_large.cuda(self.gpu)
            scaled_pads = (int(self.pad*self.fh/float(self.h)), int(self.pad*self.fw/float(self.w)))
            padded_test_input_large = F.pad(padded_test_input_large, (scaled_pads[1],scaled_pads[1],scaled_pads[0],scaled_pads[0]), mode='reflect')
            
            V_m = list()
            V_f = list()
            old_correpondence = None
            for t in range(self.TM):
                sys.stdout.write("\rProcessing frame %d, " % (t+1))
                sys.stdout.flush()
                
                flow = P_m(test_input, z_m)
                flow[:,0,:,:] = flow[:,0,:,:]*(self.w/float(self.pad*2+self.w))
                flow[:,1,:,:] = flow[:,1,:,:]*(self.h/float(self.pad*2+self.h))
                flow = F.pad(flow, (self.pad,self.pad,self.pad,self.pad), mode='reflect')
                flow = self.s_m*flow
                correspondence = initial_coordinate + flow
    
                if old_correpondence is not None:
                    correspondence = F.grid_sample(old_correpondence, correspondence.permute(0,2,3,1), padding_mode='border')
    
                correspondence_large = F.upsample(correspondence, size=(self.fh+scaled_pads[0]*2,self.fw+scaled_pads[1]*2), mode='bilinear', align_corners=True)
                y_large = F.grid_sample(padded_test_input_large, correspondence_large.permute(0,2,3,1), padding_mode='border')
                outimg = y_large.data.cpu().numpy()[0].transpose(1,2,0)
                outimg = denormalize(outimg)
                outimg = outimg[scaled_pads[0]:outimg.shape[0]-scaled_pads[0],scaled_pads[1]:outimg.shape[1]-scaled_pads[1]]
                V_m.append(outimg)
                
                outflowimg = flow.data.cpu().numpy()[0].transpose(1,2,0)
                outflowimg = outflowimg[self.pad:outflowimg.shape[0]-self.pad,self.pad:outflowimg.shape[1]-self.pad]
                mag, ang = cv2.cartToPolar(outflowimg[...,1], outflowimg[...,0])
                hsv = np.zeros_like(test_img)
                hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = 255
                outflowimg = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                outflowimg = cv2.resize(outflowimg,(self.fw,self.fh))
                V_f.append(outflowimg)
                
                y = F.grid_sample(padded_test_input, correspondence.permute(0,2,3,1), padding_mode='border')
                test_input = y[:,:,self.pad:y.shape[2]-self.pad,self.pad:y.shape[3]-self.pad]
                old_correpondence = correspondence
                
            V_mloop = generateLoop(V_m)
            
        return V_mloop, V_f
    
    def PredictAppearance(self, V_mloop):
        print('\nAppearance: ',)
        minimum_loop_num = int(1/self.s_a)
        P_a = ConditionalAppearanceNet(8)
        param = torch.load(self.model_path + '/PANet_weight_' + self.model_epoch + '.pth')
        P_a.load_state_dict(param)
        if self.gpu>-1:
            P_a.cuda(self.gpu)
        E_a = define_E(3,8,64,which_model_netE='resnet_128',vaeLike=True)
        param = torch.load(self.model_path + '/EANet_weight_' + self.model_epoch + '.pth')
        E_a.load_state_dict(param)
        if self.gpu>-1:
            E_a.cuda(self.gpu)
    
        with torch.no_grad():  
            interpolated_za_seq = list()         
            
            input_conditional_test = cv2.resize(V_mloop[0], (128, 128))
            input_conditional_test = np.array([normalize(input_conditional_test)])
            input_conditional_test = Variable(torch.from_numpy(input_conditional_test.transpose(0,3,1,2)))
            if self.gpu>-1:
                input_conditional_test = input_conditional_test.cuda(self.gpu)
            za_input, _ = E_a(input_conditional_test)
            interpolated_za_seq.append(za_input.clone())
    
            with open(self.model_path + '/codebook_a_' + self.model_epoch + '.pkl', 'rb') as f:
                codebook_a = pickle.load(f) if sys.version_info[0] == 2 else pickle.load(f, encoding='latin1')
            za_seq = codebook_a[int((len(codebook_a)-1)*self.t_a)]
            za_seq = [torch.from_numpy(np.array([za])) for za in za_seq]
            if self.gpu>-1:
                za_seq = [za.cuda(self.gpu) for za in za_seq]
            start_fid = None
            min_dist = float('inf')
            for t, mu in enumerate(za_seq):
                dist = F.mse_loss(za_input,mu).cpu().numpy()
                if dist < min_dist:
                    min_dist = dist
                    start_fid = t
    
            TA = len(za_seq)
            loop_num = max(minimum_loop_num, int(np.ceil(len(za_seq)/float(len(V_mloop)))))
            interpolation_size = int((loop_num*len(V_mloop)-TA)/TA)
            za1 = za_input.clone()
            for t in range(start_fid+1,TA):
                za2 = za_seq[t]
                for ti in range(interpolation_size):
                    lambd = (ti+1)/float(interpolation_size+1)
                    z = (1.-lambd)*za1 + lambd*za2
                    interpolated_za_seq.append(z)
                interpolated_za_seq.append(za2)
                za1 = za2
        
            za1 = za_input.clone()
            for t in range(start_fid-1,-1,-1):
                za2 = za_seq[t]    
                for ti in range(interpolation_size-1,-1,-1):
                    lambd = (ti+1)/float(interpolation_size+1)
                    z = (1.-lambd)*za2 + lambd*za1
                    interpolated_za_seq.insert(0, z)
                interpolated_za_seq.insert(0,za2)
                za1 = za2
    
            loop_num = int(np.ceil(TA*(interpolation_size+1)/float(len(V_mloop))))
            interpolation_size2 = int(interpolation_size+loop_num*len(V_mloop)-TA*(interpolation_size+1))
            z_start = za_input.clone() if start_fid==0 else za_seq[0]
            z_final = za_input.clone() if start_fid==TA-1 else za_seq[-1]
            for ti in range(interpolation_size2):
                lambd = (ti+1)/float(interpolation_size2+1)
                z = (1.-lambd)*z_final + lambd*z_start
                interpolated_za_seq.append(z)
    
            zaid=(interpolation_size+1)*start_fid
            V = list()
            t = 0
            for loop in range(loop_num):
                for frame in V_mloop:
                    sys.stdout.write("\rProcessing frame %d, " % (t+1))
                    sys.stdout.flush()
                    t+=1
                    
                    test_input = cv2.resize(frame, (self.w, self.h))
                    test_input = np.array([normalize(test_input)])
                    test_input = Variable(torch.from_numpy(test_input.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        test_input = test_input.cuda(self.gpu)
                    test_input_large = np.array([normalize(frame)])
                    test_input_large = Variable(torch.from_numpy(test_input_large.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        test_input_large = test_input_large.cuda(self.gpu)
                    z = interpolated_za_seq[zaid]
                    y, al, bl = P_a(test_input, z)
                    al_large = F.upsample(al, size=(self.fh,self.fw), mode='bilinear', align_corners=True)
                    bl_large = F.upsample(bl, size=(self.fh,self.fw), mode='bilinear', align_corners=True)
                    y = F.tanh(al_large*test_input_large+bl_large)
                    outimg = y.data.cpu().numpy()[0].transpose(1,2,0)
                    V.append(denormalize(outimg))
                    zaid+=1
                    if zaid>len(interpolated_za_seq)-1:
                        zaid=0
        
        return V
        
    def GenerateVideo(self):
        V_mloop, V_f = self.PredictMotion()      
        videoWrite(V_mloop, out_path = self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[-1] + '_motion.avi')
        videoWrite(V_f, out_path = self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[-1] + '_flow.avi')
        
        V = self.PredictAppearance(V_mloop)                
        videoWrite(V, out_path = self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[-1] + '.avi')
        print('\nDone.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnimatingLandscape')
    parser.add_argument('--model_path', default='./models')
    parser.add_argument('--model_epoch', default='5000')
    parser.add_argument('--gpu', default=-1)
    parser.add_argument('--input', '-i', default='./inputs/1.png')
    parser.add_argument('--motion_latent_code', '-mz', default=np.random.rand())
    parser.add_argument('--motion_speed', '-ms', default=0.2)
    parser.add_argument('--appearance_latent_code', '-az', default=np.random.rand())
    parser.add_argument('--appearance_speed', '-as', default=0.1)
    parser.add_argument('--motion_frame_number', '-mn', default=199)
    parser.add_argument('--outdir', '-o', default='./outputs')
    args = parser.parse_args()
    
    AS = AnimatingLandscape(args)
    AS.GenerateVideo()
    