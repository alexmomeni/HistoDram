import  numpy as np
import os
import  matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config
config = Config()
plt.switch_backend('agg')

def plot_glimpses(config, glimpse_images, pred_labels, probs, sampled_loc,
          X, labels, file_name ,fontsize=5, keep=True, step = None):
    N, H, W, C      = X.shape
    n_glimpses      = config.num_glimpses


    for i in range(N):
        
        os.makedirs(file_name + '/step_%s/image_%s' % (step,i))        
        cur_loc = sampled_loc[i,:,:]
        cur_loc_norm = ((cur_loc + 1)*config.input_shape/2.).astype(int)


        for glimpseI in range(n_glimpses):
            
            if probs == []:
                fig, cur_ax = plt.subplots(1,2)
            else:
                fig, cur_ax = plt.subplots(1,3)
                
            cur_ax[0].imshow(X[i])
            cur_ax[0].set_title('Image %s' %i)
            cur_ax[0].set_axis_off()
            
            glimpses = glimpse_images[i, glimpseI]
            glimpses = np.squeeze(glimpses).astype(int)
            cur_ax[1].imshow(glimpses)
            cur_ax[1].set_title('Glimpse %s' %glimpseI)
            cur_ax[1].set_axis_off()
            
            if probs != []:
                if glimpseI < n_glimpses:
                    cur_ax[2].bar(range(config.num_classes), probs[i,glimpseI,:], color='b')
                    cur_ax[2].set_aspect(3)
                    cur_ax[2].set_ylim((0,1))
                    cur_ax[2].set_xticks( range(config.num_classes))
                    cur_ax[2].set_xlabel('Digit')
                    cur_ax[2].set_title('Probabilities')

                if np.argmax(probs[i,glimpseI,:]) == labels[i]:
                    color = 'limegreen'
                else:
                    color = 'r'

            else: # use predicted labels
                if pred_labels[i] == labels[i]:
                    color = 'limegreen'
                else:
                    color = 'r'

            if glimpseI == n_glimpses-1:
                linewidth = 6.0
            else:
                linewidth = 3.0

            if keep:
                alpha_step = 1.0/n_glimpses
                alpha      = 0.0
                for h in range(glimpseI+1):
                    alpha += alpha_step
                    alpha = min(alpha,1.0)
                    add_glimpses(config, axis=cur_ax[0], loc=cur_loc_norm[h,:], color=color, linewidth=linewidth, alpha=alpha)
                    
                    add_glimpses(config, axis=cur_ax[1], loc=[config.glimpse_size/2,config.glimpse_size/2], color=color, linewidth=7, alpha= 0.75)
                              
            png_name = file_name + '/step_%s/image_%s/glimpse_%s.png' % (step,i,glimpseI)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.savefig(png_name, bbox_inches='tight')
            plt.close()

    return



def plot_trajectories(config=None, locations=[],
                  X = [], labels=[], pred_labels=[],
                  grid =  [], file_name = [], bboxes=[], fontsize=5, alpha=0.5, step = None):

    N, H, W, C  = X.shape
    n_glimpses  = len(locations)

    hw = W/2.

    if grid != []:
        assert grid[0]*grid[1] == N

    if grid == []:
        n_rows = int(np.ceil(np.sqrt(N)))
        n_cols = n_rows
    else:
        n_rows,n_cols = grid

    fig, ax = plt.subplots(n_rows,n_cols)
    print(ax)

    pixel_locations = ((locations[:,:-1,:] + 1)*config.input_shape/2.0).astype(int)
    
    for i,cur_ax in enumerate(ax.flat):
        if i < N:
            
            cur_ax.imshow(X[i])
            cur_ax.set_axis_off()
            
            if len(labels) != 0:
                cur_ax.set_title(('Truth %s | Pred %s') %(labels[i], pred_labels[i]))

            cur_locations = np.squeeze(pixel_locations[i,:, :]) # all locations for current image
            
            if pred_labels[i] == labels[i]:
                color = 'limegreen'
            else:
                color = 'r'

            cur_ax.plot(cur_locations[:,1],cur_locations[:,0],'-',color=color,linewidth=1.5, alpha=alpha)
            cur_ax.scatter(cur_locations[0,1],cur_locations[0,0],15, facecolors='none', linewidth=1.5, color=color, alpha=alpha)
            cur_ax.plot(cur_locations[-1,1],cur_locations[-1,0],'o',color=color,markersize=5, alpha=alpha)
            
            if bboxes != []:
                bbox = create_bbox([(bboxes[i,1]+1)*hw,(bboxes[i,0]+1)*hw,bboxes[i,2],bboxes[i,3]],
                                   color=[1,1,1], alpha=0.3, linewidth=0.7)
                cur_ax.add_patch(bbox)
        
        else:
            fig.delaxes(cur_ax)

    plt.tight_layout(pad=0.2)

    if file_name == []:
        plt.show()
        
    else:
        png_name = file_name + '/step_%s.png' % step
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(png_name, bbox_inches='tight')
        plt.close()
    return


def add_glimpses(config, axis, loc, color='r', linewidth=1.5, alpha=1):
    w =  config.glimpse_size
    hw = w/2.0
    box = [int(loc[1]-hw), int(loc[0]-hw), w, w]
    axis.add_patch(create_bbox(box, color=color, linewidth=linewidth, alpha=alpha))
    return

def create_bbox(bb, color = 'red', linewidth=1.5, alpha=1.0):
    return mpatches.Rectangle((bb[0],bb[1]),bb[2],bb[3],
                          edgecolor=color, fill=False, linewidth=linewidth, alpha=alpha)  

def norm2ind(norm_ind, width):

    return np.round(width*( (norm_ind+1)/2.0),1)

def ind2norm(ind, width):
    return np.round((ind/float(width))*2-1,1)