import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import utils
import nibabel as nib
import torch.nn.functional as F



class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,save_folder,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None,Mirror_data=False):
        if logger is None:
            self.logger = utils.get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.save_folder=save_folder
        self.Mirror_data=Mirror_data
        #self.train_folder=train_folder
        self.wt_path='/lila/data/deasy/Eric_Data/3D_Unet/'+save_folder+'/'#+residual_seg_acc.csv'   
        self.result_sv_path=self.wt_path+'result_save/'
        #path_tep='/lila/data/deasy/Eric_Data/3D_Unet/'+save_folder
        if not (os.path.isdir(self.wt_path)):
            os.mkdir(self.wt_path)
        if not (os.path.isdir(self.result_sv_path)):
            os.mkdir(self.result_sv_path)            
        self.fd_results = open(self.wt_path+'residual_seg_acc.csv', 'w')
        self.fd_results.write('Parotid_L,Parotid_R,Submand_L,Submand_R,Mandible,Cord,BrainStem,Oral_Cav,Larynx,Chiasm,OptNrv_L,OptNrv_R,Eye_L,Eye_R,\n')    
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.acc_all_previous=np.zeros(14,dtype='double')
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        logger=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]

        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        logger=None):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1


    def normalize_data(self,data):
        #index_nan=np.argwhere(np.isnan(data))
        #index_inf=np.argwhere(np.isneginf(data))
        #if len(index_nan)>0:
        #    print (len(index_nan))
        data[data<24]=24
        data[data>1524]=1524
        
        data=data-24
        
        #data=data*2./1500 - 1
        #data=data*1./1500
        return  (data)

    def fit_3D(self):

        if 1>2:
        # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
        # inifinite examples! Don't do "for batch in tr_gen:"!!!):
            num_batches_per_epoch = 10
            num_validation_batches_per_epoch = 3
            num_epochs = 5
            # let's run this to get a time on how long it takes
            time_per_epoch = []
            start = time()
            for epoch in range(num_epochs):
                start_epoch = time()
                for b in range(num_batches_per_epoch):
                    batch = next(tr_gen)
                    # do network training here with this batch

                
                end_epoch = time()
                time_per_epoch.append(end_epoch - start_epoch)
            end = time()
            total_time = end - start
            print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
                (num_epochs, total_time, str(time_per_epoch)))

        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()
        num_epochs=100
        num_batches_per_epoch=3000
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.Mirror_data:
            print ('warning: Mirror_data!')
        for epoch in range(num_epochs):
            i=0
            for b in range(num_batches_per_epoch):
                i=i+1
                t = next(self.loaders['train_loader'])
            
                #for i, t in enumerate(self.loaders['train_loader']):
                #print ('i is ', i)
                #print ('t is ', t)
                #self.logger.info(
                #    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

                # Get the data
                #print (t)
                #print (t.size())
                # Here is the process the data
                weight=None
                #print (t)
                input=t['data']
                input=self.normalize_data(input)

                #print (input.shape) [b,n,x,y,z]
                target=t['seg']

                input=np.transpose(input,(0,1,3,2,4))
                target=np.transpose(target,(0,1,3,2,4))            
                
                input=np.flip(input, 3)  # flip around y-axis
                target=np.flip(target, 3) # flip around y-axis

                # start mirror the data 
                #Mirror_data=False
                if self.Mirror_data:
                    tep_t=np.random.random() 
                    if tep_t<0.5:
                        input=np.flip(input, 2)
                        target=np.flip(target, 2)


                        ## 1-2   3-4   5-6
                        target_tep=target.copy() 
                        target[target<7]=0

                        target[target_tep==1]=2
                        target[target_tep==2]=1
                        target[target_tep==3]=4
                        target[target_tep==4]=3
                        target[target_tep==5]=6
                        target[target_tep==6]=5


                target[target>11]=0
                #print ('target sz is ',target.shape)
                input=torch.from_numpy(input.copy()).float().to(device)
                target=torch.from_numpy(target.copy()).float().to(device)
                b,_,x,y,z=target.size()
                target=target.view(b,x,y,z)

                #input, target, weight = self._split_training_batch(t)

                #input=torch.cat((input,input),0)
                #target=torch.cat((target,target),0)
                #input=torch.cat((input1,input),0)
                #target=torch.cat((target1,target),0)            
                #print (input.size())
                #print (target.size())
                #print (weight)
                #print (input)
                #print ('*'*50)
                #print (target)
                output, loss = self._forward_pass(input, target, weight)
                if i%500==0:
                    if 1>0:
                        if 1>0:
                            input_=input[0,:,:,:,:]
                            #print ('input max is ',input.max())
                            #print ('input min is ',input.min())
                            #print (input_.size())
                            #print (input)
                            aa_=input_.size()
                            #print (aa_)
                            input_=input_.view(aa_[1],aa_[2],aa_[3])
                            #print ('intput shape ',input_.size())
                            input_save=input_.data.cpu().numpy()
                            input_save=input_save.astype(np.int16)#=
                            #input_save=np.transpose(input_save,(2,1,0))
                            #input_save=np.transpose(input_save,(0,2,1))
                            #print ('2222222222222222222')    

                            #print ('val out shape is ',out_save.shape)
                            #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                            #print ('target shape is OK',target.shape)
                            
                            out_save=target[0]
                            #print (target.size())
                            out_seg_save=torch.argmax(output,1)    
                            out_seg_save=out_seg_save[0]
                            #out_save=out_save.view(out_save.size(1),out_save.size(2),out_save.size(3))     
                            #print ('1111111111111111111')           
                            out_save=out_save.data.cpu().numpy()
                            out_save=out_save.astype(np.int16)#=
                            out_save=out_save.reshape(out_save.shape[0],out_save.shape[1],out_save.shape[2])
                            
                            out_seg_save=out_seg_save.data.cpu().numpy()
                            out_seg_save=out_seg_save.astype(np.int16)#=
                            out_seg_save=out_seg_save.reshape(out_seg_save.shape[0],out_seg_save.shape[1],out_seg_save.shape[2])
                            
                            #out_save=np.transpose(out_save,(2,1,0))
                            #out_save=np.transpose(out_save,(0,2,1))
                            #print ('2222222222222222222')    
                            #print ('out_save shape is OK',out_save.shape)
                            #print ('val out shape is ',out_save.shape)
                            out_save = nib.Nifti1Image(out_save,np.eye(4))    
                            out_seg_save = nib.Nifti1Image(out_seg_save,np.eye(4))    
                            in_save = nib.Nifti1Image(input_save,np.eye(4))   
                            #print ('33333333333333333333')     
                            #out_save.get_data_dtype() == np.dtype(np.int8)        
                            #print ('4444444444444444444444')          
                            # save the seg_msk
                            val_save_name=self.result_sv_path+'training_GT_'+str(i+1)+'.nii'
                            val_img_save_name=self.result_sv_path+'training_IMG_'+str(i+1)+'.nii'
                            val_seg_save_name=self.result_sv_path+'training_SEG_'+str(i+1)+'.nii'
                            #print ('save name is ',val_save_name)
                            nib.save(out_save, val_save_name)
                            nib.save(in_save, val_img_save_name)
                            nib.save(out_seg_save, val_seg_save_name)

                #if i %200==0:
                    #self.logger.info(
                    #            f'Training stats. Loss: {loss}')
                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if 2>1:
                    if self.num_iterations % self.validate_after_iters == 0:
                    #if 3>2:
                        # evaluate on validation set
                        eval_score = self.validate_3D_PDDCA(self.loaders['val_loader'])
                        # adjust learning rate if necessary
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(eval_score)
                        else:
                            self.scheduler.step()
                        # log current learning rate in tensorboard
                        self._log_lr()
                        # remember best validation metric
                        is_best = self._is_best_eval_score(eval_score)

                        # save checkpoint
                        self._save_checkpoint(is_best)

                    if self.num_iterations % self.log_after_iters == 0:
                        # if model contains final_activation layer for normalizing logits apply it, otherwise both
                        # the evaluation metric as well as images in tensorboard will be incorrectly computed
                        if hasattr(self.model, 'final_activation'):
                            output = self.model.final_activation(output)

                        # compute eval criterion
                        #print ('output is ',output.size())
                        #print ('target is ',target.size())
                        target=target.long()
                        tr_sz=target.size()
                        #print (output.size())
                        #print (target.size())
                        #target=target.view(tr_sz[0],1,tr_sz[1],tr_sz[2],tr_sz[3])
                        eval_score =0# self.eval_criterion(output, target)
                        train_eval_scores.update(0, self._batch_size(input))

                        # log stats, params and images
                        self.logger.info(
                            f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                        self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                        #self._log_params()
                    #self._log_images(input, target, output)

                if self.max_num_iterations < self.num_iterations:
                    self.logger.info(
                        f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                    return True

                self.num_iterations += 1



    def fit_3D_newdata(self):

        if 1>2:
        # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
        # inifinite examples! Don't do "for batch in tr_gen:"!!!):
            num_batches_per_epoch = 10
            num_validation_batches_per_epoch = 3
            num_epochs = 5
            # let's run this to get a time on how long it takes
            time_per_epoch = []
            start = time()
            for epoch in range(num_epochs):
                start_epoch = time()
                for b in range(num_batches_per_epoch):
                    batch = next(tr_gen)
                    # do network training here with this batch

                
                end_epoch = time()
                time_per_epoch.append(end_epoch - start_epoch)
            end = time()
            total_time = end - start
            print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
                (num_epochs, total_time, str(time_per_epoch)))

        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()
        num_epochs=100
        num_batches_per_epoch=1000
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.Mirror_data:
            print ('warning: Mirror_data!')
        for epoch in range(num_epochs):
            i=0
            #for b in range(2):
            for b in range(num_batches_per_epoch):
                i=i+1
                t = next(self.loaders['train_loader'])
            
                #for i, t in enumerate(self.loaders['train_loader']):
                #print ('i is ', i)
                #print ('t is ', t)
                #self.logger.info(
                #    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

                # Get the data
                #print (t)
                #print (t.size())
                # Here is the process the data
                weight=None
                #print (t)
                input=t['data']
                input=self.normalize_data(input)

                #print (input.shape) [b,n,x,y,z]
                target=t['seg']

                input=np.transpose(input,(0,1,3,2,4))
                target=np.transpose(target,(0,1,3,2,4))            
                
                input=np.flip(input, 3)  # flip around y-axis
                target=np.flip(target, 3) # flip around y-axis

                # start mirror the data 
                #Mirror_data=False


                target[target>11]=0

                #Here do this is for the validation data 1/2  3/4 5/6 flip label
                target_tep_tep=target.copy() 
                target[target<7]=0

                target[target_tep_tep==1]=2
                target[target_tep_tep==2]=1
                target[target_tep_tep==3]=4
                target[target_tep_tep==4]=3
                target[target_tep_tep==5]=6
                target[target_tep_tep==6]=5

                if self.Mirror_data:
                    tep_t=np.random.random() 
                    if tep_t<0.5:
                        input=np.flip(input, 2)
                        target=np.flip(target, 2)


                        ## 1-2   3-4   5-6
                        target_tep=target.copy() 
                        target[target<7]=0

                        target[target_tep==1]=2
                        target[target_tep==2]=1
                        target[target_tep==3]=4
                        target[target_tep==4]=3
                        target[target_tep==5]=6
                        target[target_tep==6]=5


                
                #print ('target sz is ',target.shape)
                input=torch.from_numpy(input.copy()).float().to(device)
                target=torch.from_numpy(target.copy()).float().to(device)
                b,_,x,y,z=target.size()
                target=target.view(b,x,y,z)

                #input, target, weight = self._split_training_batch(t)

                #input=torch.cat((input,input),0)
                #target=torch.cat((target,target),0)
                #input=torch.cat((input1,input),0)
                #target=torch.cat((target1,target),0)            
                #print (input.size())
                #print (target.size())
                #print (weight)
                #print (input)
                #print ('*'*50)
                #print (target)
                output, loss = self._forward_pass(input, target, weight)
                if i%500==0:
                    if 1>0:
                        if 1>0:
                            input_=input[0,:,:,:,:]
                            #print ('input max is ',input.max())
                            #print ('input min is ',input.min())
                            #print (input_.size())
                            #print (input)
                            aa_=input_.size()
                            #print (aa_)
                            input_=input_.view(aa_[1],aa_[2],aa_[3])
                            #print ('intput shape ',input_.size())
                            input_save=input_.data.cpu().numpy()
                            input_save=input_save.astype(np.int16)#=
                            #input_save=np.transpose(input_save,(2,1,0))
                            #input_save=np.transpose(input_save,(0,2,1))
                            #print ('2222222222222222222')    

                            #print ('val out shape is ',out_save.shape)
                            #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                            #print ('target shape is OK',target.shape)
                            
                            out_save=target[0]
                            #print (target.size())
                            out_seg_save=torch.argmax(output,1)    
                            out_seg_save=out_seg_save[0]
                            #out_save=out_save.view(out_save.size(1),out_save.size(2),out_save.size(3))     
                            #print ('1111111111111111111')           
                            out_save=out_save.data.cpu().numpy()
                            out_save=out_save.astype(np.int16)#=
                            out_save=out_save.reshape(out_save.shape[0],out_save.shape[1],out_save.shape[2])
                            
                            out_seg_save=out_seg_save.data.cpu().numpy()
                            out_seg_save=out_seg_save.astype(np.int16)#=
                            out_seg_save=out_seg_save.reshape(out_seg_save.shape[0],out_seg_save.shape[1],out_seg_save.shape[2])
                            
                            #out_save=np.transpose(out_save,(2,1,0))
                            #out_save=np.transpose(out_save,(0,2,1))
                            #print ('2222222222222222222')    
                            #print ('out_save shape is OK',out_save.shape)
                            #print ('val out shape is ',out_save.shape)
                            out_save = nib.Nifti1Image(out_save,np.eye(4))    
                            out_seg_save = nib.Nifti1Image(out_seg_save,np.eye(4))    
                            in_save = nib.Nifti1Image(input_save,np.eye(4))   
                            #print ('33333333333333333333')     
                            #out_save.get_data_dtype() == np.dtype(np.int8)        
                            #print ('4444444444444444444444')          
                            # save the seg_msk
                            val_save_name=self.result_sv_path+'training_GT_'+str(i+1)+'.nii'
                            val_img_save_name=self.result_sv_path+'training_IMG_'+str(i+1)+'.nii'
                            val_seg_save_name=self.result_sv_path+'training_SEG_'+str(i+1)+'.nii'
                            #print ('save name is ',val_save_name)
                            nib.save(out_save, val_save_name)
                            nib.save(in_save, val_img_save_name)
                            nib.save(out_seg_save, val_seg_save_name)

                #if i %200==0:
                    #self.logger.info(
                    #            f'Training stats. Loss: {loss}')
                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if 2>1:
                    if self.num_iterations % self.validate_after_iters == 0:
                    #if 3>2:
                        # evaluate on validation set
                        eval_score = self.validate_3D_Newdata(self.loaders['val_loader'])
                        # adjust learning rate if necessary
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(eval_score)
                        else:
                            self.scheduler.step()
                        # log current learning rate in tensorboard
                        self._log_lr()
                        # remember best validation metric
                        is_best = self._is_best_eval_score(eval_score)

                        # save checkpoint
                        self._save_checkpoint(is_best)

                    if self.num_iterations % self.log_after_iters == 0:
                        # if model contains final_activation layer for normalizing logits apply it, otherwise both
                        # the evaluation metric as well as images in tensorboard will be incorrectly computed
                        if hasattr(self.model, 'final_activation'):
                            output = self.model.final_activation(output)

                        # compute eval criterion
                        #print ('output is ',output.size())
                        #print ('target is ',target.size())
                        target=target.long()
                        tr_sz=target.size()
                        #print (output.size())
                        #print (target.size())
                        #target=target.view(tr_sz[0],1,tr_sz[1],tr_sz[2],tr_sz[3])
                        eval_score =0# self.eval_criterion(output, target)
                        train_eval_scores.update(0, self._batch_size(input))

                        # log stats, params and images
                        self.logger.info(
                            f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                        self._log_stats('train', train_losses.avg, train_eval_scores.avg)

                    if self.num_iterations % (5*self.log_after_iters) == 0:
                    #if 1>0:
                        train_loss_dsc=self.cal_dice_loss_print(output.detach().cpu(),target.detach().cpu())
                        self.logger.info(
                            f'LP: {train_loss_dsc[0]}. RP: {train_loss_dsc[1]}. LS: {train_loss_dsc[2]}. RS: {train_loss_dsc[3]}. Lpix: {train_loss_dsc[4]}. Rpix: {train_loss_dsc[5]}')
                        self.logger.info(
                            f'Man: {train_loss_dsc[6]}. SP: {train_loss_dsc[7]}. BS: {train_loss_dsc[8]}. Oral_cav: {train_loss_dsc[9]}. Larynx: {train_loss_dsc[10]}')
                    #self._log_images(input, target, output)

                if self.max_num_iterations < self.num_iterations:
                    self.logger.info(
                        f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                    return True

                self.num_iterations += 1
            if epoch%10==0:
                self._save_checkpoint_epoch(epoch)


    def cal_dice_loss_print(self,pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        """
        t_b,t_x,t_y,t_z=target.size()
        
        #print ('dd ',pred_stage1.shape)
        pred_stage1=torch.argmax(pred_stage1,dim=1)
        #torch.argmax(output,1)   
        
            # loss
        dice_0=0
        dice_1=0
        dice_2=0
        dice_3=0
        dice_4=0
        dice_5=0
        dice_6=0
        dice_7=0
        dice_8=0
        dice_9=0
        dice_10=0
        dice_11=0
        dice_12=0

        dice_stage1 = 0.0   
        smooth = 1.
        for organ_index in  range(1,12):
            pred_tep=torch.zeros((target.size(0), t_x,t_y,t_z))
            target_tep=torch.zeros((target.size(0) , t_x,t_y,t_z))
            #print (pred_stage1.shape)
            #print (organ_target.shape)
            pred_tep[pred_stage1==organ_index]=1#pred_stage1[:, organ_index,  :, :,:] 
            target_tep[target==organ_index]=1#organ_target[:, organ_index, :, :,:] 
            #pred_tep=pred_stage1[:, 0,  :, :]   # move back
            #target_tep=organ_target[:, 0,  :, :] # move back


            #print (pred_tep.size())
            #print (target.size())
            pred_tep=pred_tep.contiguous().view(-1)
            target_tep=target_tep.contiguous().view(-1)
            #print (pred_tep)
            #print (target_tep)
            intersection_tp = (pred_tep * target_tep).sum()
            dice_tp=(2. * intersection_tp + smooth)/(pred_tep.sum() + target_tep.sum() + smooth)
            
            
            if organ_index==1:
                dice_1=dice_tp
            if organ_index==2:
                dice_2=dice_tp
            if organ_index==3:
                dice_3=dice_tp
            if organ_index==4:
                dice_4=dice_tp
            if organ_index==5:
                dice_5=dice_tp
            if organ_index==6:
                dice_6=dice_tp
            if organ_index==7:
                dice_7=dice_tp
            if organ_index==8:
                dice_8=dice_tp
            if organ_index==9:
                dice_9=dice_tp
            if organ_index==10:
                dice_10=dice_tp
            if organ_index==11:
                dice_11=dice_tp

            





        #print (dice_0)
        #print (dice_1)
        #print (dice_7)
        return dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8,dice_9,dice_10,dice_11

    def cal_dice_loss_print_bake(self,pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        """
        t_b,t_x,t_y,t_z=target.size()
        organ_target = torch.zeros((target.size(0),12 , t_x,t_y,t_z))  # 8+1
        #print ('dd ',pred_stage1.shape)
        pred_stage1=F.softmax(pred_stage1,dim=1)

        for organ_index in range(12):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            #print (temp_target.shape)
            #print (organ_target[:, organ_index, :, :].shape)
            organ_target[:, organ_index,  :, :] = temp_target.reshape(temp_target.shape[0],t_x,t_y,t_z)
            # organ_target: (B, 8,  128, 128)

        organ_target = organ_target.cpu()

            # loss
        dice_0=0
        dice_1=0
        dice_2=0
        dice_3=0
        dice_4=0
        dice_5=0
        dice_6=0
        dice_7=0
        dice_8=0
        dice_9=0
        dice_10=0
        dice_11=0
        dice_12=0

        dice_stage1 = 0.0   
        smooth = 1.
        for organ_index in  range(1,12):
            #print (pred_stage1.shape)
            #print (organ_target.shape)
            pred_tep=pred_stage1[:, organ_index,  :, :,:] 
            target_tep=organ_target[:, organ_index, :, :,:] 
            #pred_tep=pred_stage1[:, 0,  :, :]   # move back
            #target_tep=organ_target[:, 0,  :, :] # move back


            #print (pred_tep.size())
            #print (target.size())
            pred_tep=pred_tep.contiguous().view(-1)
            target_tep=target_tep.contiguous().view(-1)
            #print (pred_tep)
            #print (target_tep)
            intersection_tp = (pred_tep * target_tep).sum()
            dice_tp=(2. * intersection_tp + smooth)/(pred_tep.sum() + target_tep.sum() + smooth)
            
            
            if organ_index==1:
                dice_1=dice_tp
            if organ_index==2:
                dice_2=dice_tp
            if organ_index==3:
                dice_3=dice_tp
            if organ_index==4:
                dice_4=dice_tp
            if organ_index==5:
                dice_5=dice_tp
            if organ_index==6:
                dice_6=dice_tp
            if organ_index==7:
                dice_7=dice_tp
            if organ_index==8:
                dice_8=dice_tp
            if organ_index==9:
                dice_9=dice_tp
            if organ_index==10:
                dice_10=dice_tp
            if organ_index==11:
                dice_11=dice_tp

            





        #print (dice_0)
        #print (dice_1)
        #print (dice_7)
        return dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8,dice_9,dice_10,dice_11

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            #print ('i is ', i)
            #print ('t is ', t)
            #self.logger.info(
            #    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            # Get the data
            input, target, weight = self._split_training_batch(t)

            #input=torch.cat((input,input),0)
            #target=torch.cat((target,target),0)
            #input=torch.cat((input1,input),0)
            #target=torch.cat((target1,target),0)            
            #print (input.size())
            #print (target.size())
            #print (weight)
            output, loss = self._forward_pass(input, target, weight)
            if i==1:
                if 1>0:
                    if 1>0:
                        aa_=input.size()
                        input=input.view(aa_[2],aa_[3],aa_[4])
                        input_save=input.data.cpu().numpy()
                        input_save=input_save.astype(np.int16)#=
                        #input_save=np.transpose(input_save,(2,1,0))
                        #input_save=np.transpose(input_save,(0,2,1))
                        #print ('2222222222222222222')    

                        #print ('val out shape is ',out_save.shape)
                        #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                        #print ('target shape is OK',target.shape)
                        
                        out_save=target
                        #out_save=torch.argmax(output,1)    
                        #out_save=out_save.view(out_save.size(1),out_save.size(2),out_save.size(3))     
                        #print ('1111111111111111111')           
                        out_save=out_save.data.cpu().numpy()
                        out_save=out_save.astype(np.int16)#=
                        out_save=out_save.reshape(out_save.shape[1],out_save.shape[2],out_save.shape[3])
                        #out_save=np.transpose(out_save,(2,1,0))
                        #out_save=np.transpose(out_save,(0,2,1))
                        #print ('2222222222222222222')    
                        #print ('out_save shape is OK',out_save.shape)
                        #print ('val out shape is ',out_save.shape)
                        out_save = nib.Nifti1Image(out_save,np.eye(4))    
                        in_save = nib.Nifti1Image(input_save,np.eye(4))   
                        #print ('33333333333333333333')     
                        #out_save.get_data_dtype() == np.dtype(np.int8)        
                        #print ('4444444444444444444444')          
                        # save the seg_msk
                        val_save_name='/lila/data/deasy/Eric_Data/3D_Unet/GT_'+str(i+1)+str(i+1)+'.nii'
                        val_img_save_name='/lila/data/deasy/Eric_Data/3D_Unet/img_'+str(i+1)+str(i+1)+'.nii'
                        #print ('save name is ',val_save_name)
                        #nib.save(out_save, val_save_name)
                        #nib.save(in_save, val_img_save_name)

            #if i %200==0:
                #self.logger.info(
                #            f'Training stats. Loss: {loss}')
            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if 2>1:
                if self.num_iterations % self.validate_after_iters == 0:
                    # evaluate on validation set
                    eval_score = self.validate(self.loaders['val'])
                    # adjust learning rate if necessary
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(eval_score)
                    else:
                        self.scheduler.step()
                    # log current learning rate in tensorboard
                    self._log_lr()
                    # remember best validation metric
                    is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                    self._save_checkpoint(is_best)

                if self.num_iterations % self.log_after_iters == 0:
                    # if model contains final_activation layer for normalizing logits apply it, otherwise both
                    # the evaluation metric as well as images in tensorboard will be incorrectly computed
                    if hasattr(self.model, 'final_activation'):
                        output = self.model.final_activation(output)

                    # compute eval criterion
                    #print ('output is ',output.size())
                    #print ('target is ',target.size())
                    target=target.long()
                    eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                    # log stats, params and images
                    self.logger.info(
                        f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                    self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                    self._log_params()
                #self._log_images(input, target, output)

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        return False


    def validate_3D(self, val_loader):
        self.logger.info('Validating...')
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        num_batches_per_epoch=10
        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            acc_all=np.zeros(14,dtype='double')
            self.model.eval()
            with torch.no_grad():
                val_iter=0
                val_iter=0
                i=0
                for b in range(num_batches_per_epoch):
                    t = next(val_loader)
                    i=i+1
                    
                    if 1>0:
                    
                        print (i)
                        val_iter=val_iter+1
                        
                        
                        #self.logger.info(f'Validation iteration {i}')
                        if i<20:

                            
                            weight=None
                            #print (t)
                            input=t['data']
                            input=self.normalize_data(input)
                            target=t['seg']

                            input=np.transpose(input,(0,1,3,2,4))
                            target=np.transpose(target,(0,1,3,2,4))            
                            
                            input=np.flip(input, 3)
                            target=np.flip(target, 3)


                            target[target>9]=0
                            input=torch.from_numpy(input.copy()).float().to(device)
                            target=torch.from_numpy(target.copy()).float().to(device)
                            b,_,x,y,z=target.size()
                            target=target.view(b,x,y,z)

                            output, loss = self._forward_pass(input, target, weight)
                            output, loss = self._forward_pass(input, target, weight)
                            output_1_48,loss=self._forward_pass(input[:,:,:,:,0:48], target[:,:,:,0:48], weight)
                            output_30_78,loss=self._forward_pass(input[:,:,:,:,30:78], target[:,:,:,30:78], weight)
                            
                            output[:,:,:,:,0:48]=output_1_48
                            output[:,:,:,:,30:78]=output_30_78                            
                            #print ('input size',input.size())
                            #print (output)
                            validation_=True
                            if validation_:
                                aa_=input.size()
                                target_=target.view(aa_[2],aa_[3],aa_[4])
                                input=input.view(aa_[2],aa_[3],aa_[4])
                                input_save=input.data.cpu().numpy()
                                target_=target_.data.cpu().numpy()
                                input_save=input_save.astype(np.int16)#=
                                target_=target_.astype(np.int16)#=
                                #input_save=np.transpose(input_save,(2,1,0))
                                #input_save=np.transpose(input_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',output.shape)
                                #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                                out_result=torch.argmax(output,1)   
                                #print ('val out_save shape is ',out_save.shape) 
                                out_save=out_result.view(out_result.size()[1],out_result.size()[2],out_result.size()[3])     
                                #print ('1111111111111111111')           
                                out_save=out_save.data.cpu().numpy()
                                out_result=out_save
                                out_save=out_save.astype(np.int16)#=
                                #out_save=np.transpose(out_save,(2,1,0))
                                #out_save=np.transpose(out_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',out_save.shape)
                                out_save = nib.Nifti1Image(out_save,np.eye(4))    
                                in_save = nib.Nifti1Image(input_save,np.eye(4))   
                                target_ = nib.Nifti1Image(target_,np.eye(4))   

                                #print ('33333333333333333333')     
                                #out_save.get_data_dtype() == np.dtype(np.int8)        
                                #print ('4444444444444444444444')          
                                # save the seg_msk
                                val_save_name=self.result_sv_path+'seg_'+str(i+1)+'.nii'
                                val_img_save_name=self.result_sv_path+'img_'+str(i+1)+'.nii'
                                val_target_save_name=self.result_sv_path+'gt_'+str(i+1)+'.nii'
                                #print ('save name is ',val_save_name)
                                nib.save(out_save, val_save_name)
                                nib.save(in_save, val_img_save_name)
                                nib.save(target_, val_target_save_name)
                                #print ('55555555555555555')    


                                # calcualte and report the segmentation accuracy

                                for cal_id in range(1,15):
                                    # seg:out_result
                                    # gt: target
                                    target_numpy=target.data.cpu().numpy()
                                    gt_check=np.zeros(target_numpy.shape,dtype=target_numpy.dtype)
                                    seg_check=np.zeros(out_result.shape,dtype=target_numpy.dtype)
                                    seg_check[out_result==cal_id]=1
                                    gt_check[target_numpy==cal_id]=1

                                    acc_=self.cal_3D_dice(seg_check,gt_check)
                                    acc_all[cal_id-1]=acc_all[cal_id-1]+acc_
                                #acc_all=acc_all/10


                            
                            val_losses.update(loss.item(), self._batch_size(input))
                            target=target.long()
                            eval_score = self.eval_criterion(output, target)
                            val_scores.update(eval_score.item(), self._batch_size(input))

                            if self.validate_iters is not None and self.validate_iters <= i:
                                # stop validation
                                break
                acc_all=acc_all/20.

                self.fd_results.write(str(acc_all[0] ) + ',' +str(acc_all[1] ) + ','+str(acc_all[2] ) + ','+str(acc_all[3] ) + ','+str(acc_all[4] ) + ','+str(acc_all[5] ) + ','
                        +str(acc_all[6] ) + ','+str(acc_all[7] ) + ','+str(acc_all[8] ) + ','+str(acc_all[9] ) + ','+str(acc_all[10] ) + ','+str(acc_all[11] ) + ','
                        +str(acc_all[12] ) + ','+str(acc_all[13] ) +'\n')
                self.fd_results.flush()                         

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                save_best=True
                return val_scores.avg

            
        finally:
            # set back in training mode
            self.model.train()

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            acc_all=np.zeros(14,dtype='double')
            self.model.eval()
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    #self.logger.info(f'Validation iteration {i}')

                    
                    input, target, weight = self._split_training_batch(t)

                    output, loss = self._forward_pass(input, target, weight)
                    #print ('input size',input.size())
                    #print (output)
                    validation_=True
                    if validation_:
                        aa_=input.size()
                        target_=target.view(aa_[2],aa_[3],aa_[4])
                        input=input.view(aa_[2],aa_[3],aa_[4])
                        input_save=input.data.cpu().numpy()
                        target_=target_.data.cpu().numpy()
                        input_save=input_save.astype(np.int16)#=
                        target_=target_.astype(np.int16)#=
                        #input_save=np.transpose(input_save,(2,1,0))
                        #input_save=np.transpose(input_save,(0,2,1))
                        #print ('2222222222222222222')    

                        #print ('val out shape is ',output.shape)
                        #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                        out_result=torch.argmax(output,1)   
                        #print ('val out_save shape is ',out_save.shape) 
                        out_save=out_result.view(out_result.size()[1],out_result.size()[2],out_result.size()[3])     
                        #print ('1111111111111111111')           
                        out_save=out_save.data.cpu().numpy()
                        out_result=out_save
                        out_save=out_save.astype(np.int16)#=
                        #out_save=np.transpose(out_save,(2,1,0))
                        #out_save=np.transpose(out_save,(0,2,1))
                        #print ('2222222222222222222')    

                        #print ('val out shape is ',out_save.shape)
                        out_save = nib.Nifti1Image(out_save,np.eye(4))    
                        in_save = nib.Nifti1Image(input_save,np.eye(4))   
                        target_ = nib.Nifti1Image(target_,np.eye(4))   

                        #print ('33333333333333333333')     
                        #out_save.get_data_dtype() == np.dtype(np.int8)        
                        #print ('4444444444444444444444')          
                        # save the seg_msk
                        val_save_name=self.result_sv_path+'seg_'+str(i+1)+'.nii'
                        val_img_save_name=self.result_sv_path+'img_'+str(i+1)+'.nii'
                        val_target_save_name=self.result_sv_path+'gt_'+str(i+1)+'.nii'
                        #print ('save name is ',val_save_name)
                        nib.save(out_save, val_save_name)
                        nib.save(in_save, val_img_save_name)
                        nib.save(target_, val_target_save_name)
                        #print ('55555555555555555')    


                        # calcualte and report the segmentation accuracy

                        for cal_id in range(1,15):
                            # seg:out_result
                            # gt: target
                            target_numpy=target.data.cpu().numpy()
                            gt_check=np.zeros(target_numpy.shape,dtype=target_numpy.dtype)
                            seg_check=np.zeros(out_result.shape,dtype=target_numpy.dtype)
                            seg_check[out_result==cal_id]=1
                            gt_check[target_numpy==cal_id]=1

                            acc_=self.cal_3D_dice(seg_check,gt_check)
                            acc_all[cal_id-1]=acc_all[cal_id-1]+acc_
                        #acc_all=acc_all/10


                    
                    val_losses.update(loss.item(), self._batch_size(input))
                    eval_score = self.eval_criterion(output, target)
                    val_scores.update(eval_score.item(), self._batch_size(input))

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break
                acc_all=acc_all/10.

                self.fd_results.write(str(acc_all[0] ) + ',' +str(acc_all[1] ) + ','+str(acc_all[2] ) + ','+str(acc_all[3] ) + ','+str(acc_all[4] ) + ','+str(acc_all[5] ) + ','
                        +str(acc_all[6] ) + ','+str(acc_all[7] ) + ','+str(acc_all[8] ) + ','+str(acc_all[9] ) + ','+str(acc_all[10] ) + ','+str(acc_all[11] ) + ','
                        +str(acc_all[12] ) + ','+str(acc_all[13] ) +'\n')
                self.fd_results.flush()                         

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                save_best=True

            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    #self.logger.info(f'Validation iteration {i}')

                    
                    input, target, weight = self._split_training_batch(t)

                    output, loss = self._forward_pass(input, target, weight)                

                    if save_best:
                        if save_best:
                            if i>-100:
                                input, target, weight = self._split_training_batch(t)
                                aa_=input.size()
                                target_=target.view(aa_[2],aa_[3],aa_[4])
                                input=input.view(aa_[2],aa_[3],aa_[4])
                                
                                out_result=torch.argmax(output,1)   
                                #print ('val out_save shape is ',out_save.shape) 
                                out_save=out_result.view(out_result.size()[1],out_result.size()[2],out_result.size()[3])     
                                #print ('1111111111111111111')           
                                out_save=out_save.data.cpu().numpy()
                                out_result=out_save
                                out_save=out_save.astype(np.int16)#=

                                out_save = nib.Nifti1Image(out_save,np.eye(4))    


                                
                                for cal_id in range(1,15):
                                    if acc_all[cal_id-1]>=self.acc_all_previous[cal_id-1]:
                                        val_save_name=self.result_sv_path+'seg_'+str(i+1)+'_best_'+str(cal_id)+'.nii'
                                        nib.save(out_save, val_save_name)
                            #print ('save name is ',val_save_name)
                            

                self.acc_all_previous=acc_all
                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)
        #target=torch.cat((target,target[:,:,:,0:32,0:32]),2)
        #target=torch.cat((target,targetx[:,:,:,0:32,0:32]),3)
        
        #target=torch.tensor(np.ndarray(shape=(1,32,128,128), dtype=float, order='F'))
        #target.fill(0)
        target=target.cuda().long()
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _save_checkpoint_epoch(self, is_best):
        utils.save_checkpoint_epoch({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)




    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
           # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / np.ptp(img)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    @staticmethod 
    def cal_3D_dice(seg_tep,gt_tep):
        smooth=1.
        seg_flt=seg_tep.flatten()
        gt_flt=gt_tep.flatten()
                                
        intersection = np.sum(seg_flt * gt_flt)
        dsc_3D_tep=(2. * intersection + smooth) / (np.sum(seg_flt) + np.sum(gt_flt) + smooth)
        return dsc_3D_tep



    def validate_3D_PDDCA(self, val_loader):
        self.logger.info('Validating...')
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        num_batches_per_epoch=10
        print ('validating!!!!!!!!!!!')
        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            acc_all=np.zeros(14,dtype='double')
            ct_all=np.zeros(14,dtype='double')
            self.model.eval()
            with torch.no_grad():
                val_iter=0
                val_iter=0
                i=0
                for b, t in enumerate(zip(val_loader)):
                    #t = next(val_loader)
                    i=i+1
                    
                    if 1>0:
                    
                        #print (i)
                        val_iter=val_iter+1
                        
                        
                        #self.logger.info(f'Validation iteration {i}')
                        if 1>0:

                            
                            weight=None
                            t=t[0]
                            #print (t)
                            input=t['data']
                            input=self.normalize_data(input)
                            target=t['seg']

                            input=np.transpose(input,(0,1,3,2,4))
                            target=np.transpose(target,(0,1,3,2,4))            
                            
                            input=np.flip(input, 3)
                            target=np.flip(target, 3)


                            target[target>9]=0
                            input=torch.from_numpy(input.copy()).float().to(device)
                            target=torch.from_numpy(target.copy()).float().to(device)
                            b,_,x,y,z=target.size()
                            target=target.view(b,x,y,z)

                            output, loss = self._forward_pass(input, target, weight)
                            output_1_48,loss=self._forward_pass(input[:,:,:,:,0:48], target[:,:,:,0:48], weight)
							
                            output_30_78,loss=self._forward_pass(input[:,:,:,:,z-48:z], target[:,:,:,z-48:z], weight)
                            
                            output[:,:,:,:,0:48]=output_1_48
                            output[:,:,:,:,z-48:z]=output_30_78
                            
                            
                            #print ('input size',input.size())
                            #print (output)
                            validation_=True
                            if validation_:
                                aa_=input.size()
                                target_=target.view(aa_[2],aa_[3],aa_[4])
                                input=input.view(aa_[2],aa_[3],aa_[4])
                                input_save=input.data.cpu().numpy()
                                target_=target_.data.cpu().numpy()
                                input_save=input_save.astype(np.int16)#=
                                target_=target_.astype(np.int16)#=
                                #input_save=np.transpose(input_save,(2,1,0))
                                #input_save=np.transpose(input_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',output.shape)
                                #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                                out_result=torch.argmax(output,1)   
                                #print ('val out_save shape is ',out_save.shape) 
                                out_save=out_result.view(out_result.size()[1],out_result.size()[2],out_result.size()[3])     
                                #print ('1111111111111111111')           
                                out_save=out_save.data.cpu().numpy()
                                out_result=out_save
                                out_save=out_save.astype(np.int16)#=
                                #out_save=np.transpose(out_save,(2,1,0))
                                #out_save=np.transpose(out_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',out_save.shape)
                                out_save = nib.Nifti1Image(out_save,np.eye(4))    
                                in_save = nib.Nifti1Image(input_save,np.eye(4))   
                                target_ = nib.Nifti1Image(target_,np.eye(4))   

                                #print ('33333333333333333333')     
                                #out_save.get_data_dtype() == np.dtype(np.int8)        
                                #print ('4444444444444444444444')          
                                # save the seg_msk
                                val_save_name=self.result_sv_path+'seg_'+str(i+1)+'.nii'
                                val_img_save_name=self.result_sv_path+'img_'+str(i+1)+'.nii'
                                val_target_save_name=self.result_sv_path+'gt_'+str(i+1)+'.nii'
                                #print ('save name is ',val_save_name)
                                nib.save(out_save, val_save_name)
                                nib.save(in_save, val_img_save_name)
                                nib.save(target_, val_target_save_name)
                                #print ('55555555555555555')    


                                # calcualte and report the segmentation accuracy

                                for cal_id in range(1,15):
                                    # seg:out_result
                                    # gt: target
                                    target_numpy=target.data.cpu().numpy()
                                    gt_check=np.zeros(target_numpy.shape,dtype=target_numpy.dtype)
                                    seg_check=np.zeros(out_result.shape,dtype=target_numpy.dtype)
                                    seg_check[out_result==cal_id]=1
                                    gt_check[target_numpy==cal_id]=1

                                    if np.max(gt_check)>0:
                                        acc_=self.cal_3D_dice(seg_check,gt_check)
                                        acc_all[cal_id-1]=acc_all[cal_id-1]+acc_
                                        ct_all[cal_id-1]=ct_all[cal_id-1]+1
                                #acc_all=acc_all/10


                            
                            val_losses.update(loss.item(), self._batch_size(input))
                            target=target.long()
                            eval_score = self.eval_criterion(output, target)
                            val_scores.update(eval_score.item(), self._batch_size(input))

                            if self.validate_iters is not None and self.validate_iters <= i:
                                # stop validation
                                break
                #acc_all=acc_all/48.
                acc_all[0]= acc_all[0]/ ct_all[0]
                acc_all[1]= acc_all[1]/ ct_all[1]
                acc_all[2]= acc_all[2]/ ct_all[2]
                acc_all[3]= acc_all[3]/ ct_all[3]
                acc_all[6]= acc_all[6]/ ct_all[6]
                acc_all[8]= acc_all[8]/ ct_all[8]

                self.fd_results.write(str(acc_all[0] ) + ',' +str(acc_all[1] ) + ','+str(acc_all[2] ) + ','+str(acc_all[3] ) + ','+str(acc_all[4] ) + ','+str(acc_all[5] ) + ','
                        +str(acc_all[6] ) + ','+str(acc_all[7] ) + ','+str(acc_all[8] ) + ','+str(acc_all[9] ) + ','+str(acc_all[10] ) + ','+str(acc_all[11] ) + ','
                        +str(acc_all[12] ) + ','+str(acc_all[13] ) +'\n')
                print ('LP val accuracy is:', acc_all[0])   
                print ('RP val accuracy is:', acc_all[1])   
                print ('LS val accuracy is:', acc_all[2])   
                print ('RS val accuracy is:', acc_all[3])   
                print ('Man val accuracy is:', acc_all[6])  
                print ('BS val accuracy is:', acc_all[8])   
                
                self.fd_results.flush()                         

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                #save_best=True
                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()

    def validate_3D_Newdata(self, val_loader):
        self.logger.info('Validating...')
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        num_batches_per_epoch=10
        print ('validating!!!!!!!!!!!')
        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            acc_all=np.zeros(14,dtype='double')
            ct_all=np.zeros(14,dtype='double')
            dsc_3d_sv=np.zeros((53,11),dtype='double')
            #self.model.eval()
            with torch.no_grad():
                val_iter=0
                val_iter=0
                i=0
                for b_tch_id, t in enumerate(zip(val_loader)):
                    #t = next(val_loader)
                    i=i+1
                    
                    if 1>0:
                    
                        #print (i)
                        val_iter=val_iter+1
                        
                        
                        #self.logger.info(f'Validation iteration {i}')
                        if 1>0:

                            
                            weight=None
                            t=t[0]
                            #print (t)
                            input=t['data']
                            input=self.normalize_data(input)
                            target=t['seg']

                            input=np.transpose(input,(0,1,3,2,4))
                            target=np.transpose(target,(0,1,3,2,4))            
                            
                            input=np.flip(input, 3)
                            target=np.flip(target, 3)


                            target[target>11]=0
                            input=torch.from_numpy(input.copy()).float().to(device)
                            target=torch.from_numpy(target.copy()).float().to(device)
                            b,_,x,y,z=target.size()
                            target=target.view(b,x,y,z)

                            output, loss = self._forward_pass(input, target, weight)
                            #output_1_48,loss=self._forward_pass(input[:,:,:,:,0:48], target[:,:,:,0:48], weight)
							
                            #output_30_78,loss=self._forward_pass(input[:,:,:,:,z-48:z], target[:,:,:,z-48:z], weight)
                            
                            #output[:,:,:,:,0:48]=output_1_48
                            #output[:,:,:,:,z-48:z]=output_30_78
                            
                            
                            #print ('input size',input.size())
                            #print (output)
                            validation_=True
                            if validation_:
                                aa_=input.size()
                                target_=target.view(aa_[2],aa_[3],aa_[4])
                                input=input.view(aa_[2],aa_[3],aa_[4])
                                input_save=input.data.cpu().numpy()
                                target_=target_.data.cpu().numpy()
                                input_save=input_save.astype(np.int16)#=
                                target_=target_.astype(np.int16)#=
                                #input_save=np.transpose(input_save,(2,1,0))
                                #input_save=np.transpose(input_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',output.shape)
                                #input_save = nib.Nifti1Image(out_input_savesave,np.eye(4))                           

                                out_result=torch.argmax(output,1)   
                                #print ('val out_save shape is ',out_save.shape) 
                                out_save=out_result.view(out_result.size()[1],out_result.size()[2],out_result.size()[3])     
                                #print ('1111111111111111111')           
                                out_save=out_save.data.cpu().numpy()
                                out_result=out_save
                                out_save=out_save.astype(np.int16)#=
                                #out_save=np.transpose(out_save,(2,1,0))
                                #out_save=np.transpose(out_save,(0,2,1))
                                #print ('2222222222222222222')    

                                #print ('val out shape is ',out_save.shape)
                                if (i%10)==0:
                                    out_save = nib.Nifti1Image(out_save,np.eye(4))    
                                    in_save = nib.Nifti1Image(input_save,np.eye(4))   
                                    target_ = nib.Nifti1Image(target_,np.eye(4))   

                                    #print ('33333333333333333333')     
                                    #out_save.get_data_dtype() == np.dtype(np.int8)        
                                    #print ('4444444444444444444444')          
                                    # save the seg_msk
                                    val_save_name=self.result_sv_path+'seg_'+str(i+1)+'.nii'
                                    val_img_save_name=self.result_sv_path+'img_'+str(i+1)+'.nii'
                                    val_target_save_name=self.result_sv_path+'gt_'+str(i+1)+'.nii'
                                    #print ('save name is ',val_save_name)
                                    nib.save(out_save, val_save_name)
                                    nib.save(in_save, val_img_save_name)
                                    nib.save(target_, val_target_save_name)
                                #print ('55555555555555555')    


                                # calcualte and report the segmentation accuracy

                                for cal_id in range(1,12):
                                    # seg:out_result
                                    # gt: target
                                    target_numpy=target.data.cpu().numpy()
                                    gt_check=np.zeros(target_numpy.shape,dtype=target_numpy.dtype)
                                    seg_check=np.zeros(out_result.shape,dtype=target_numpy.dtype)
                                    seg_check[out_result==cal_id]=1
                                    gt_check[target_numpy==cal_id]=1
                                    
                                    #if 1>0:
                                    if np.max(gt_check)>0:
                                        acc_=self.cal_3D_dice(seg_check,gt_check)
                                        acc_all[cal_id-1]=acc_all[cal_id-1]+acc_
                                        ct_all[cal_id-1]=ct_all[cal_id-1]+1
                                        dsc_3d_sv[b_tch_id,cal_id-1]=acc_
                                    else:
                                        dsc_3d_sv[b_tch_id,cal_id-1]=-100
                                    
                                
                                #acc_all=acc_all/10


                            
                            val_losses.update(loss.item(), self._batch_size(input))
                            target=target.long()
                            eval_score = self.eval_criterion(output, target)
                            val_scores.update(eval_score.item(), self._batch_size(input))

                            if self.validate_iters is not None and self.validate_iters <= i:
                                # stop validation
                                break
                dsc_txt_sv_name=self.wt_path+'dsc_3d_all.txt'
                np.savetxt(dsc_txt_sv_name, dsc_3d_sv,fmt='%1.6f')
                #acc_all=acc_all/48.
                print (ct_all)
                acc_all[0]= acc_all[0]/ ct_all[0]
                acc_all[1]= acc_all[1]/ ct_all[1]
                acc_all[2]= acc_all[2]/ ct_all[2]
                acc_all[3]= acc_all[3]/ ct_all[3]
                acc_all[4]= acc_all[4]/ ct_all[4]
                acc_all[5]= acc_all[5]/ ct_all[5]
                acc_all[6]= acc_all[6]/ ct_all[6]
                acc_all[7]= acc_all[7]/ ct_all[7]
                acc_all[8]= acc_all[8]/ ct_all[8]
                acc_all[9]= acc_all[9]/ ct_all[9]
                acc_all[10]= acc_all[10]/ ct_all[10]
                #acc_all[11]= acc_all[10]/ ct_all[11]

                #self.fd_results.write(str(acc_all[0] ) + ',' +str(acc_all[1] ) + ','+str(acc_all[2] ) + ','+str(acc_all[3] ) + ','+str(acc_all[4] ) + ','+str(acc_all[5] ) + ','
                #        +str(acc_all[6] ) + ','+str(acc_all[7] ) + ','+str(acc_all[8] ) + ','+str(acc_all[9] ) + ','+str(acc_all[10] ) + ','+str(acc_all[11] ) + ','
                #        +str(acc_all[12] ) + ','+str(acc_all[13] ) +'\n')
                print ('LP val accuracy is:', acc_all[0])   
                print ('RP val accuracy is:', acc_all[1])   
                print ('LS val accuracy is:', acc_all[2])   
                print ('RS val accuracy is:', acc_all[3]) 
                print ('L_Pixvuel val accuracy is:', acc_all[4])
                print ('R_Pixvuel val accuracy is:', acc_all[5])  
                print ('Man val accuracy is:', acc_all[6])  
                print ('SPCord val accuracy is:', acc_all[7])  
                print ('BS val accuracy is:', acc_all[8])   
                print ('Oral_Cavl accuracy is:', acc_all[9])  
                print ('Larynx val accuracy is:', acc_all[10])  

                self.fd_results.write(str(acc_all[0] ) + ',' +str(acc_all[1] ) + ','+str(acc_all[2] ) + ','+str(acc_all[3] ) + ','+str(acc_all[4] ) + ','+str(acc_all[5] ) + ','
                        +str(acc_all[6] ) + ','+str(acc_all[7] ) + ','+str(acc_all[8] ) + ','+str(acc_all[9] ) + ','+str(acc_all[10] ) + ','+str(acc_all[11] ) + ','
                        +str(acc_all[12] ) + ','+str(acc_all[13] ) +'\n')
                
                
                self.fd_results.flush()                         

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                save_best=True
                return val_scores.avg


        finally:
            # set back in training mode
            self.model.train()