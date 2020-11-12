import os
import sys
import cv2
import numpy as np
import glob

BOARD_FILL_COLOR = 1e-5

def main():

    benchmark='./data'
    
    
    datadir_train=benchmark+'/train'
    datadir_test =benchmark+'/test'
    if not os.path.exists('./data/'):
        os.mkdir('./data')
    if not os.path.exists('./data/ndata_seq/'):
        os.mkdir('./data/ndata_seq/')
    if not os.path.exists('./data/ndata_single/'):
        os.mkdir('./data/ndata_single/')
    if not os.path.exists('./data/nlabel'):
        os.mkdir('./data/nlabel')
    # if not os.path.exists('./data/ndata_full'):
    #     os.mkdir('./data/ndata_full')
    # if not os.path.exists('./data/nlabel_full'):
    #     os.mkdir('./data/nlabel_full')
    seq_train, seq_test=glob.glob('./train/*'), glob.glob('./test/*')
    seq_all=seq_train+seq_test
    #generate npy data
    for i,seq in enumerate(seq_all):
        #illu for a sequence
        seq_id=seq.split('/')[-1]
        marker='train' if i<len(seq_train) else 'test'
        gt_file=os.path.join(seq, 'groundtruth.txt')
        with open(gt_file, 'r') as f:
            illu=f.readline()
        illu=illu.strip().split(','); illu=list(map(float,illu))
        #imgs for a sequence
        files_seq=glob.glob(seq+'\\[0-9]*.png')
        files_seq.sort(key=lambda x:x[:-4].split('/')[-1])
        img_list=[]
        for file in files_seq:
            file_id=file[:-4].split('/')[-1]
            raw=np.array(cv2.imread(file, -1), dtype='float32')
            img_list.append(raw)
            
        seq_id=seq_id.split('\\')[-1]    
            

        np.save('./data/ndata_seq/' + marker+seq_id +'.npy', img_list)
        np.save('./data/ndata_single/'+ marker+seq_id +'.npy', img_list[-1])
        np.save('./data/nlabel/'+ marker+seq_id +'.npy', illu)
        # for j,img in enumerate(img_list):
        #     np.save('./data/ndata_full/' + marker+'_'+seq_id+'_' +str(j)+'.npy', img)
        #     np.save('./data/nlabel_full/' + marker+'_'+seq_id+'_' +str(j)+'.npy', illu)

        print(i)




def load_image_without_mcc(fn,mcc_coord):
    raw = load_image(fn)
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.float32) # clip constrain the value between 0 and 1
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]]) #the vertex of polygon
    polygon = polygon.astype(np.int32)
    cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3) # fill the polygon to img
    return img

def load_image(fn):
    file_path = './data/images/' + fn
    raw = np.array(cv2.imread(file_path, -1), dtype='float32')
    #print(raw)
    if fn.startswith('IMG'):
      # 5D3 images
      black_point = 129
    else:
      black_point = 1
    raw = np.maximum(raw - black_point, [0, 0, 0])  # remain the pixels that raw-black_point>0
    return raw

def get_mcc_coord(fn):
    # Note: relative coord
    with open('./data/coordinates/' + fn.split('.')[0] +'_macbeth.txt', 'r') as f:
        lines = f.readlines()
        width, height = map(float, lines[0].split())
        scale_x = 1 / width
        scale_y = 1 / height
        lines = [lines[1], lines[2], lines[4], lines[3]]
        polygon = []
        for line in lines:
            line = line.strip().split()
            x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
            polygon.append((x, y))
        return np.array(polygon, dtype='float32')

if __name__=='__main__':
    main()
