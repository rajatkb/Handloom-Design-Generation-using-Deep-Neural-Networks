import cv2
import numpy as np
import os


def parse_folder(folder, augment=False):
    i = 1
    train_data=[]
    print("reading "+folder+" !!!")
    for root , subFolder , files in os.walk(folder):
        try:
            for file in files:
                if file.split('.')[1] in ('jpg','png','jpeg'):
                    img = cv2.imread(root+'/'+file)
                    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                    train_data.append(img)
                    if augment:
                        train_data.append(np.flip(img,0))
                        train_data.append(np.flip(img,1))
                        train_data.append(np.flip(np.flip(img,0),1))
                        train_data.append(np.flip(np.flip(img,1),0))
                        cim = np.flip(img,2)
                        train_data.append(np.flip(cim,0))
                        train_data.append(np.flip(cim,1))
                        train_data.append(np.flip(np.flip(cim,0),1))
                        train_data.append(np.flip(np.flip(cim,1),0))
                    print('processed image: ',i)
                    i = i + 1
                    # if i > 1:
                    #     break
        except OSError as e:
            print('failed at file: ',root+'/'+file,' with ', e)
    return np.array(train_data)

if __name__=='__main__':  
    import argparse
    parser = argparse.ArgumentParser(description='data builder')
    parser.add_argument('--tfx', action="store",dest="target_folderX", default='data/X' , required=True)
    parser.add_argument('--tfy', action="store",dest="target_folderY", default='data/Y' , required=True)
    parser.add_argument('--augx', action="store",dest="augmentx", default=False)
    parser.add_argument('--augy', action="store",dest="augmenty", default=False)
    values = parser.parse_args()
    
    target_folderX = values.target_folderX
    target_folderY = values.target_folderY
    augmentx = values.augmentx
    augmenty = values.augmenty
    
    X = parse_folder(target_folderX , augmentx)
    Y = parse_folder(target_folderY , augmenty)
    print("X :",X.shape , " Y :" , Y.shape)
    np.save('X.npy',X)
    np.save('Y.npy',Y)
    


