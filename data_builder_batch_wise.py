import cv2
import numpy as np
import os

## python data_builder.py --tfx saree_data --augx True --resolution 256

def parse_folder(folder, augment=False , resolution=512 , batch=-1):
    image_count = 1
    batch_count = 1
    train_data=[]
    try:
        if(batch > 0):
            os.mkdir("numpy_data")
    except FileExistsError as e:
        print("Folder exist !! Moving on")
    print("reading "+folder+" !!!")
    for root , subFolder , files in os.walk(folder):
        try:
            for file in files:
                if file.split('.')[1] in ('jpg','png','jpeg'):
                    img = cv2.imread(root+'/'+file)
                    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img , (resolution , resolution))
                    train_data.append(img)
                    if augment == True:
                        train_data.append(np.flip(img,0))
                        train_data.append(np.flip(img,1))
                        train_data.append(np.flip(np.flip(img,0),1))
                        train_data.append(np.flip(np.flip(img,1),0))
                        cim = np.flip(img,2)
                        train_data.append(np.flip(cim,0))
                        train_data.append(np.flip(cim,1))
                        train_data.append(np.flip(np.flip(cim,0),1))
                        train_data.append(np.flip(np.flip(cim,1),0))
                    print('processed image: ',image_count)
                    image_count += 1
                    if(image_count == batch*batch_count):
                        print(" saving batch at numpy_data: batch_",batch_count," data samples: ",len(train_data),sep='')
                        np.save("numpy_data/batch_"+str(batch_count)+".npy" , np.array(train_data))
                        batch_count+=1
                        train_data = []

        except OSError as e:
            print('failed at file: ',root+'/'+file,' with error: ', e)

    print(" saving batch at numpy_data: batch_",batch_count)
    np.save("numpy_data/batch_"+str(batch_count)+".npy" , np.array(train_data))
    
    return None

if __name__=='__main__':  
    import argparse
    parser = argparse.ArgumentParser(description='data builder')
    parser.add_argument('--t', action="store",dest="target_folder", default='data' , required=True)
    # parser.add_argument('--tfy', action="store",dest="target_folderY", default='data/Y' , required=True)
    parser.add_argument('--aug', action="store",dest="augment", default=False)
    # parser.add_argument('--augy', action="store",dest="augmenty", default=False)
    parser.add_argument('--res',action="store",dest="resolution",required = True)
    parser.add_argument('--batch',action="store",dest="batch",default=-1)
    values = parser.parse_args()
    batch = int(values.batch)
    target_folder = values.target_folder
    # target_folderY = values.target_folderY
    augment = values.augment
    # augmenty = values.augmenty
    resolution = int(values.resolution)
    parse_folder(target_folder , augment , resolution , batch)
    # Y = parse_folder(target_folderY , augmenty , resolution)
    # np.save('Y.npy',Y)
    


