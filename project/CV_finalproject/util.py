import numpy as np
import re
import sys
import cv2
def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[
        2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)
    
def cal_avgerr(GT, disp):
    return np.sum(np.multiply(np.abs(GT - disp), GT[GT != np.inf])) / np.sum(GT[GT != np.inf])

def main():
    print('[Bad Pixel Ratio]')
#    max_disp = 60
    scale_factor = 4
    
    for i in range(10):
        avg = 0
#        .png
        disp =  cv2.imread('TS'+str(i)+'.png')
        disp = cv2.cvtColor(disp,cv2.COLOR_BGR2GRAY).astype(int)
#        .pfm
#        disp =  readPFM('TS'+str(i)+'.pfm')
        
        
        h, w = disp.shape
        disp = np.reshape(disp,(1,h*w))
        gt = np.reshape(readPFM('./data/Synthetic/TLD'+str(i)+'.pfm'),(1,h*w))
        
        avg+=cal_avgerr(gt*scale_factor, disp)
        print('TLD'+str(i)+': %.2f%%' % (avg/10))


#    cv2.imwrite('1.png', GT*scale_factor)

if __name__ == '__main__':
    main()