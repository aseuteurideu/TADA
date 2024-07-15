import torch
import glob


import os


def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False, save_last_epoch=True):
    if save_last_epoch:
        torch.save(state, filename)
        last_epoch_path = os.path.join(os.path.dirname(filename),
                                       'epoch%s.pth.tar' % str(state['epoch'] - gap))
        if (state['epoch'] - gap) == 50:  # keep the 50th epoch result. change the path. Move it to halfepochs_results
            os.makedirs(os.path.join(os.path.dirname(filename), 'halfepochs_results'), exist_ok=True)
            os.rename(last_epoch_path,
                      os.path.join(os.path.dirname(filename), 'halfepochs_results', 'epoch%s.pth.tar' % str(state['epoch'] - gap)))
            past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
            for i in past_best:
                os.rename(i,
                          os.path.join(os.path.dirname(filename), 'halfepochs_results', os.path.basename(i)))
        if not keep_all:
            try:
                os.remove(last_epoch_path)
            except:
                pass

    if is_best:
        if 'late' in filename:
            prefix = 'late'
        elif 'mid' in filename:
            prefix = 'mid'
        elif 'early' in filename:
            prefix = 'early'
        else:
            prefix = ''

        past_best = glob.glob(os.path.join(os.path.dirname(filename), prefix + 'model_best_*.pth.tar'))
        for i in past_best:
            try:
                os.remove(i)
            except:
                pass
        torch.save(state, os.path.join(os.path.dirname(filename), prefix + 'model_best_epoch%s.pth.tar' % str(state['epoch'])))

###############################################################################################
def our_attack(data, epsilon, data_grad):
    # epsilon positive for to the same direction of the gradient (if the data_grad will minimize the loss)
    # epsilon negative for to the opposite direction of the gradient (if the data_grad will maximize the loss)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data - epsilon * sign_data_grad
    # Adding clipping to maintain [-1,1] range
    perturbed_data = torch.clamp(perturbed_data, -1, 1)
    # Return the perturbed image
    return perturbed_data

