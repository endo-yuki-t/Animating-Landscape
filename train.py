# encoding: utf-8

from __future__ import print_function
import argparse,glob, re, os, random, pickle
import numpy as np
import cv2
import util
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from predictor import ConditionalMotionNet, ConditionalAppearanceNet, GramMatrix, Discriminator
from encoder import define_E
from vgg16 import Vgg16

class TrainAnimatingLandscape():
    
    def __init__(self, args):
        self.out_dir = args.outdir
        self.gpu = int(args.gpu)
        self.frame_interval = 2 #Used to omit pairs of distant frames from learning
        self.unchanged_pair_threshold = 0.02 #Used to omit less changed pairs from learning
        self.dir_list = glob.glob(args.indir + '/*')
        print("The number of training video clips:", len(self.dir_list))
        self.saved_epoch = int(args.save_epoch_freq)
        self.max_epoch = int(args.max_epoch)
        self.learning_rate = args.learning_rate
        
        self.lambda_tvm = float(args.lambda_tvm)
        self.lambda_p = float(args.lambda_pm)

        self.lambda_tva = float(args.lambda_tva)
        self.lambda_spa = float(args.lambda_spa)
        self.sppooled_size = int(args.spa_grid_size)
        self.lambda_sa = float(args.lambda_sa)
        self.lambda_ca = float(args.lambda_ca)
        
        self.lambda_WAE = float(args.lambda_wae)
        self.batch_size= int(args.batch_size)
        self.sigma = float(args.sigma)
        self.nz = int(args.nz)
        self.w, self.h = int(args.image_size), int(args.image_size)
        
    def TrainMotionModels(self):
        
        model_E = define_E(2,self.nz,64,which_model_netE='resnet_128')
        if self.gpu>-1:
            model_E.cuda(self.gpu)
        optimizer_E = Adam(model_E.parameters(),  lr=self.learning_rate, betas=(0.5, 0.999))   
    
        model_P = ConditionalMotionNet(self.nz)
        if self.gpu>-1:
            model_P.cuda(self.gpu)
        optimizer_P = Adam(model_P.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            
        if self.lambda_WAE > 0.:
            model_WAE_D = Discriminator()
            if self.gpu>-1:
                model_WAE_D.cuda(self.gpu)
            optimizer_WAE_D = Adam(model_WAE_D.parameters(),lr=self.learning_rate*0.1)
    
        initial_flow = np.array([np.meshgrid(np.linspace(-1,1,self.w/1), np.linspace(-1,1,self.h/1), sparse=False)]).astype(np.float32)
        initial_flow = Variable(torch.from_numpy(initial_flow))
        zero_flow = Variable(torch.zeros(initial_flow.shape))
        if self.gpu>-1:
            initial_flow = initial_flow.cuda(self.gpu)
            zero_flow = zero_flow.cuda(self.gpu)
        
        pre_flow_dict = dict()
        pre_codebook = dict()
        for epoch in range(1,self.max_epoch+1):
            batch_loss = 0.
            wae_loss = 0.
            sample_num = 0
            total_loss = 0.
            np.random.shuffle(self.dir_list)
            for dir in self.dir_list:   
                fnames = glob.glob(dir+"/*.jpg")
                fnames.sort(key=util.natural_keys)            
                frame_id = random.randint(0,len(fnames)-2)
                fnames = fnames[frame_id:frame_id+2]
                if int(fnames[1].split("_")[-1][:-4])-int(fnames[0].split("_")[-1][:-4]) != self.frame_interval:
                    continue
                for frame_id in range(len(fnames)-1):                  
                    frame1 = cv2.imread(fnames[frame_id])
                    frame1 = cv2.resize(frame1, (self.w,self.h))                
                    frame1 = util.normalize(frame1)     
                    frame1 = np.array([frame1])
                    frame1 = Variable(torch.from_numpy(frame1.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        frame1 = frame1.cuda(self.gpu)
    
                    frame2 = cv2.imread(fnames[frame_id+1])
                    frame2 = cv2.resize(frame2, (self.w,self.h))            
                    frame2 = util.normalize(frame2)
                    frame2 = np.array([frame2])
                    frame2 = Variable(torch.from_numpy(frame2.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        frame2 = frame2.cuda(self.gpu)
                    
                    diff_mean = torch.abs(frame1-frame2).mean()
                    if diff_mean<self.unchanged_pair_threshold:
                        continue
                 
                    "====Compute latent codes===="
                    pre_flow = pre_flow_dict.get(os.path.basename(dir), zero_flow)
                    if self.w!=128 or self.h!=128:
                        pre_flow = F.upsample(pre_flow, size=(128,128), mode='bilinear', align_corners=True)
                    mu, logvar = model_E(pre_flow)
                    
                    "====WAE loss===="
                    z_encoded = mu
                    if self.lambda_WAE>0.:
                        d_fake = model_WAE_D(z_encoded)
                        z_real = Variable(torch.randn(1, self.nz))
                        if self.gpu>-1:
                            z_real = z_real.cuda(self.gpu)
                        d_real = model_WAE_D(z_real)
                        wae_loss += -(torch.mean(d_real) - torch.mean(d_fake))
                        mu, logvar = model_E(pre_flow)
                        z_encoded = mu
                        d_fake = model_WAE_D(z_encoded)
                        batch_loss += -self.lambda_WAE*torch.mean(d_fake)                        
                    
                    "====Forward===="   
                    flow = model_P(frame1, z_encoded)
                    pre_flow_dict[os.path.basename(dir)] = flow.detach()
                    pre_codebook[os.path.basename(dir)] = z_encoded.detach().cpu().numpy()[0]
                    
                    "====TV loss===="
                    if self.lambda_tvm > 0.:         
                        dxo = frame2[:, :, :, :-1] - frame2[:, :, :, 1:]
                        dyo = frame2[:, :, :-1, :] - frame2[:, :, 1:, :]
                        dxi = flow[:, :, :, :-1] - flow[:, :, :, 1:]
                        dyi = flow[:, :, :-1, :] - flow[:, :, 1:, :]
                        dxow = torch.exp(torch.sum(-torch.abs(dxo),1)/self.sigma)
                        dyow = torch.exp(torch.sum(-torch.abs(dyo),1)/self.sigma)
                        batch_loss += self.lambda_tvm*(torch.mean(dxow*torch.abs(dxi)) + torch.mean(dyow*torch.abs(dyi)))      
                    
                    "====Reconstruction loss===="
                    flow = flow + initial_flow
                    y = F.grid_sample(frame1, flow.permute(0,2,3,1), padding_mode="border")
                    if self.lambda_p>0.:
                        batch_loss += self.lambda_p*F.mse_loss(y, frame2)
                    
                    "====Backward===="
                    sample_num+=1
                    if sample_num>self.batch_size-1:
                        if self.lambda_WAE>0.:
                            optimizer_WAE_D.zero_grad()
                            wae_loss.backward()
                            optimizer_WAE_D.step()
                            for p in model_WAE_D.parameters():
                                p.data.clamp_(-0.01, 0.01)
                            wae_loss = 0.
                            
                        optimizer_P.zero_grad()
                        optimizer_E.zero_grad()
                        batch_loss.backward()
                        optimizer_P.step()
                        optimizer_E.step()
                            
                        total_loss += batch_loss.data
                        batch_loss = 0.
                        sample_num = 0
                    
            print(epoch, total_loss)
            if epoch%self.saved_epoch==0:
                model_P.cpu()
                model_E.cpu()
                torch.save(model_P.state_dict(), self.out_dir+'/PMNet_weight_%d.pth'%epoch)
                torch.save(model_E.state_dict(), self.out_dir+'/EMNet_weight_%d.pth'%epoch)
                if self.gpu>-1:
                    model_P.cuda(self.gpu)
                    model_E.cuda(self.gpu)
                codebook = np.array(list(pre_codebook.values()))
                codebook = util.sort_motion_codebook(codebook)
                f = open(self.out_dir+"/codebook_m_%d.pkl"%epoch, "wb")
                pickle.dump(codebook,f)
                f.close()
    
    def TrainAppearanceModels(self):
        
        model_E = define_E(3,self.nz,64,which_model_netE='resnet_128')
        if self.gpu>-1:
            model_E.cuda(self.gpu)
        optimizer_E = Adam(model_E.parameters(),  lr=self.learning_rate, betas=(0.5, 0.999))
        vgg = Vgg16()
        vgg.load_state_dict(torch.load(os.path.join("./models", "vgg16_weight.pth")))
        if self.gpu>-1:
            vgg.cuda(self.gpu)
        model_P = ConditionalAppearanceNet(self.nz)
        if self.gpu>-1:
            model_P.cuda(self.gpu)
        optimizer_P = Adam(model_P.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        Gram = GramMatrix()
        
        if self.lambda_WAE > 0.:
            model_WAE_D = Discriminator()
            if self.gpu>-1:
                model_WAE_D.cuda(self.gpu)
            optimizer_WAE_D = Adam(model_WAE_D.parameters(),lr=self.learning_rate*0.1)
    
        for epoch in range(1,self.max_epoch+1):
            batch_loss = 0.
            wae_loss = 0.
            sample_num = 0
            total_loss = 0.
            np.random.shuffle(self.dir_list)
            for dir in self.dir_list:
                fnames = glob.glob(dir+"/*.jpg")
                fnames = np.random.choice(fnames,2)
                for fid in range(len(fnames)-1):
                    frame1 = cv2.imread(fnames[0])
                    frame1 = cv2.resize(frame1, (self.w,self.h))         
                    frame1 = util.normalize(frame1)
                    frame1 = np.array([frame1])
                    frame1 = Variable(torch.from_numpy(frame1.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        frame1 = frame1.cuda(self.gpu)
    
                    frame2 = cv2.imread(fnames[fid+1])
                    frame2, frame2_conditional = cv2.resize(frame2, (self.w,self.h)), cv2.resize(frame2, (128,128))
                    frame2 = util.normalize(frame2)
                    frame2 = np.array([frame2])
                    frame2 = Variable(torch.from_numpy(frame2.transpose(0,3,1,2)))
                    frame2_conditional = util.normalize(frame2_conditional)
                    frame2_conditional = np.array([frame2_conditional])
                    frame2_conditional = Variable(torch.from_numpy(frame2_conditional.transpose(0,3,1,2)))
                    if self.gpu>-1:
                        frame2 = frame2.cuda(self.gpu)
                        frame2_conditional = frame2_conditional.cuda(self.gpu)
    
                    "====Compute latent codes===="
                    mu, logvar = model_E(frame2_conditional)
                       
                    "====WAE loss===="
                    z_encoded = mu
                    if self.lambda_WAE>0.:
                        d_fake = model_WAE_D(z_encoded)
                        z_real = Variable(torch.randn(1, self.nz))
                        if self.gpu>-1:
                            z_real = z_real.cuda(self.gpu)
                        d_real = model_WAE_D(z_real)
                        wae_loss += -(torch.mean(d_real) - torch.mean(d_fake))
                        mu, logvar = model_E(frame2_conditional)
                        z_encoded = mu
                        d_fake = model_WAE_D(z_encoded)
                        batch_loss += -self.lambda_WAE*torch.mean(d_fake)       
    
                    "====Forward===="   
                    y, al, bl = model_P(frame1, z_encoded)
                    
                    "====TV loss==="
                    if self.lambda_tva > 0.:       
                        dxi = frame1[:, :, :, :-1] - frame1[:, :, :, 1:]
                        dyi = frame1[:, :, :-1, :] - frame1[:, :, 1:, :]
                        dxiw = torch.exp(torch.sum(-torch.abs(dxi),1)/self.sigma)
                        dyiw = torch.exp(torch.sum(-torch.abs(dyi),1)/self.sigma)
                        dxa = al[:, :, :, :-1] - al[:, :, :, 1:]
                        dya = al[:, :, :-1, :] - al[:, :, 1:, :]
                        dxb = bl[:, :, :, :-1] - bl[:, :, :, 1:]
                        dyb = bl[:, :, :-1, :] - bl[:, :, 1:, :]
                        batch_loss += self.lambda_tva*(torch.mean(dxiw*torch.abs(dxa)) + torch.mean(dyiw*torch.abs(dya)))
                        batch_loss += self.lambda_tva*(torch.mean(dxiw*torch.abs(dxb)) + torch.mean(dyiw*torch.abs(dyb)))
               
                    "====SP loss===="
                    if self.lambda_spa>0.:
                        pooled_y = F.avg_pool2d(y, kernel_size=(y.size(2)//self.sppooled_size, y.size(3)//self.sppooled_size))
                        pooled_output = F.avg_pool2d(frame2, kernel_size=(frame2.size(2)//self.sppooled_size, frame2.size(3)//self.sppooled_size))
                        batch_loss += self.lambda_spa*F.mse_loss(pooled_y, pooled_output)
    
                    "====Style & perceptual losses===="
                    if self.lambda_sa>0. or self.lambda_ca>0.:
                        frame1_percepual_features = vgg(frame1)
                        frame2_percepual_features = vgg(frame2)
                        y_percepual_features = vgg(y)
    
                        if self.lambda_sa>0.:
                            for pfid in range(1,4):
                                output_gram = Gram(frame2_percepual_features[pfid])
                                y_gram = Gram(y_percepual_features[pfid])
                                batch_loss += self.lambda_sa*F.mse_loss(y_gram, Variable(output_gram.data, requires_grad=False))
                        if self.lambda_ca>0.:
                            pfid = 0
                            batch_loss += self.lambda_ca*F.mse_loss(y_percepual_features[pfid], Variable(frame1_percepual_features[pfid].data, requires_grad=False))
                    
                    "====Backward===="
                    sample_num+=1
                    if sample_num>self.batch_size-1:
                        if self.lambda_WAE>0.:
                            optimizer_WAE_D.zero_grad()
                            wae_loss.backward()
                            optimizer_WAE_D.step()
                            for p in model_WAE_D.parameters():
                                p.data.clamp_(-0.01, 0.01)
                            wae_loss = 0.
                            
                        optimizer_P.zero_grad()
                        optimizer_E.zero_grad()
                        batch_loss.backward()
                        optimizer_P.step()
                        optimizer_E.step()
                            
                        total_loss += batch_loss.data
                        batch_loss = 0.
                        sample_num = 0
                    
            print(epoch, total_loss)
            if epoch%self.saved_epoch==0:
                model_P.cpu()
                model_E.cpu()
                torch.save(model_P.state_dict(), self.out_dir+'/PANet_weight_%d.pth'%epoch)
                torch.save(model_E.state_dict(), self.out_dir+'/EANet_weight_%d.pth'%epoch)
                if self.gpu>-1:
                    model_P.cuda(self.gpu)
                    model_E.cuda(self.gpu)
                with torch.no_grad():
                    codebook = []
                    for dir in self.dir_list:
                        fnames = glob.glob(dir+"/*.jpg")
                        z_seq = []
                        for fname in fnames:
                            frame_conditional = cv2.imread(fname)
                            frame_conditional = cv2.resize(frame_conditional, (128,128))
                            frame_conditional = util.normalize(frame_conditional)
                            frame_conditional = np.array([frame_conditional])
                            frame_conditional = Variable(torch.from_numpy(frame_conditional.transpose(0,3,1,2)))
                            if self.gpu>-1:
                                frame_conditional = frame_conditional.cuda(self.gpu)
                            z, _ = model_E(frame_conditional)
                            z_seq.append(z.data.cpu().numpy()[0])
    
                        codebook.append(np.array(z_seq))
                    util.sort_appearance_codebook(codebook)
                    f = open(self.out_dir+"/codebook_a_%d.pkl"%epoch, "wb")
                    pickle.dump(codebook,f)
                    f.close()
                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--mode', default='motion')
    parser.add_argument('--gpu', default=-1)
    parser.add_argument('--indir', default='./training_data/motion')
    parser.add_argument('--outdir', default='./models')
    parser.add_argument('--lambda_pm', default=1.)
    parser.add_argument('--lambda_tvm', default=1.)
    parser.add_argument('--lambda_sa', default=1.)
    parser.add_argument('--lambda_spa', default=1e-2)
    parser.add_argument('--spa_grid_size', default=32)
    parser.add_argument('--lambda_ca', default=1e-5)
    parser.add_argument('--lambda_tva', default=0.1)
    parser.add_argument('--sigma', default=0.1)
    parser.add_argument('--lambda_wae', default=0.)
    parser.add_argument('--save_epoch_freq', default=100)
    parser.add_argument('--max_epoch', default=5000)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--nz', default=8)
    parser.add_argument('--image_size', default=256)
    args = parser.parse_args()
    
    TrainAL = TrainAnimatingLandscape(args)
    if args.mode == 'motion':
        TrainAL.TrainMotionModels()
    if args.mode == 'appearance':
        TrainAL.TrainAppearanceModels()   
                