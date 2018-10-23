import cv2
import numpy as np
import os


def edge(img):
    (thresh, img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.Canny(img , 50 , 250 , edges=3 , L2gradient = True)
    (thresh, img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ind_w = img > 128
    ind_b = img < 128
    img[ind_w] = 0
    img[ind_b] = 255
    img = np.stack((img,)*3,-1)
    return img


def parse_folder(folder, destination , augment=False , resolution=512):
    if not os.path.isdir(destination):
        os.mkdir(destination)
    print("Reading from "+folder+" !!!")
    i = 0
    for root , subFolder , files in os.walk(folder):
        try:
            for file in files:
                name,extension = file.split('.')
                if  extension in ('jpg','png','jpeg'):
                    img = cv2.imread(root+'/'+file)
                    img_bw = cv2.imread(root + '/'+file , cv2.IMREAD_GRAYSCALE)
					                    
                    edge_bw = edge(img_bw)

                    img = cv2.resize(img , (resolution , resolution))
                    edge_bw = cv2.resize(edge_bw , (resolution , resolution))
                    
                    final = np.concatenate([edge_bw , img] , axis=1)
                    cv2.imwrite(destination+'/'+name+'1'+str(i)+'.'+extension , final)

                    if augment:
                        x = np.flip(img,0)
                        y = np.flip(edge_bw,0)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'2'+str(i)+'.'+extension , final)

                        x = np.flip(img,1)
                        y = np.flip(edge_bw , 1)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'3'+str(i)+'.'+extension , final)


                        x = np.flip(np.flip(img,0),1)
                        y = np.flip(np.flip(edge_bw,0),1)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'4'+str(i)+'.'+extension , final)


                        x = np.flip(np.flip(img,1),0)
                        y = np.flip(np.flip(edge_bw,1),0)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'5'+str(i)+'.'+extension , final)
                        
                        cimg = np.flip(img,2)

                        x = np.flip(cimg,0)
                        y = np.flip(edge_bw,0)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'6'+str(i)+'.'+extension , final)

                        x = np.flip(cimg,1)
                        y = np.flip(edge_bw , 1)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'7'+str(i)+'.'+extension , final)


                        x = np.flip(np.flip(cimg,0),1)
                        y = np.flip(np.flip(edge_bw,0),1)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'8'+str(i)+'.'+extension , final)


                        x = np.flip(np.flip(cimg,1),0)
                        y = np.flip(np.flip(edge_bw,1),0)
                        final = np.concatenate([y , x] , axis=1)
                        cv2.imwrite(destination+'/'+name+'9'+str(i)+'.'+extension , final)
                        
                    print('processed image: ',i)
                    i = i + 1
                    # if i > 1:
                    #     break
        except OSError as e:
            print('failed at file: ',root+'/'+file,' with ', e)
    return None

if __name__=='__main__':  
	import argparse
	parser = argparse.ArgumentParser(description='data builder')
	parser.add_argument('--tf', action="store",dest="target_folder", default='data' , required=True)
	parser.add_argument('--df', action="store",dest="destination_folder", default='destination' , required=True)
	parser.add_argument('--aug', action="store",dest="augment", default=False)
	parser.add_argument('--resolution',action="store",dest="resolution",required = True)
	values = parser.parse_args()

	target_folder = values.target_folder
	destination_folder = values.destination_folder
	augment = values.augment
	resolution = int(values.resolution)
	parse_folder(target_folder ,destination_folder ,  augment , resolution)