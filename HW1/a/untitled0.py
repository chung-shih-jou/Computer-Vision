from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema
from collections import Counter


def  joint_bilateral_filter(im,sigma_s,sigma_r,r,gray_pixel,RR,GG,BB):
        width,height = im.size
        FR = np.zeros((width,height), dtype=np.int)
        ima = Image.new('RGB',(width,height))
        FG = np.zeros((width,height), dtype=np.int)
        FB = np.zeros((width,height), dtype=np.int)
        
        
        FRrgb = np.zeros((width,height), dtype=np.int)
        imargb = Image.new('RGB',(width,height))
        FGrgb = np.zeros((width,height), dtype=np.int)
        FBrgb = np.zeros((width,height), dtype=np.int)
        
        
        RR = np.zeros((width+2*r,height+2*r), dtype=np.float)
        GG = np.zeros((width+2*r,height+2*r), dtype=np.float)
        BB = np.zeros((width+2*r,height+2*r), dtype=np.float)
        
        RR[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = R/255
        GG[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = G/255
        BB[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = B/255
        a,b = np.meshgrid(np.arange(-r,+r+1,1),np.arange(-r,+r+1,1))
        Gs = np.exp(-(a**2+b**2)/(2*sigma_s**2))
        
        gray_pixel1 = np.zeros((width+2*r,height+2*r), dtype=np.float)
        gray_pixel1[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = gray_pixel[:,:]
        for y in range(r,height+r):
                        for x in range(r,width+r):
                                a,b = np.meshgrid(np.arange(x-r,x+r+1,1),np.arange(y-r,y+r+1,1))
                                g = gray_pixel1[a[:,:],b[:,:]]
                                rr = RR[a[:,:],b[:,:]]
                                gg = GG[a[:,:],b[:,:]]
                                bb = BB[a[:,:],b[:,:]]
                                ###3 channel
                                Grrgb = np.exp(-(((rr-RR[x][y])**2+(gg-GG[x][y])**2+(bb-BB[x][y])**2)/(2*sigma_r**2)))
                                FuRrgb = sum(sum(Grrgb*Gs*rr))
                                FuGrgb = sum(sum(Grrgb*Gs*gg))
                                FuBrgb = sum(sum(Grrgb*Gs*bb))
                                Fdrgb = sum(sum(Grrgb*Gs))
                                FRrgb[x-r][y-r] = int(FuRrgb/Fdrgb*255)
                                FGrgb[x-r][y-r] = int(FuGrgb/Fdrgb*255)
                                FBrgb[x-r][y-r] = int(FuBrgb/Fdrgb*255)
                                rgba2 = (FRrgb[x-r][y-r],FGrgb[x-r][y-r],FBrgb[x-r][y-r])
                                imargb.putpixel((x-r,y-r),rgba2)
                                ###1 channel
                                Gr = np.exp(-((g-gray_pixel1[x][y])**2)/(2*sigma_r**2))
                                FuR = sum(sum(Gr*Gs*rr))
                                FuG = sum(sum(Gr*Gs*gg))
                                FuB = sum(sum(Gr*Gs*bb))
                                Fd = sum(sum(Gr*Gs))
                                FR[x-r][y-r] = int(FuR/Fd*255)
                                FG[x-r][y-r] = int(FuG/Fd*255)
                                FB[x-r][y-r] = int(FuB/Fd*255)
                                rgba = FR[x-r][y-r],FG[x-r][y-r],FB[x-r][y-r]
                                ima.putpixel((x-r,y-r),rgba)
        ima.save("JTB"+str(i)+str(j)+"_"+str(number) + ".png")
        imargb.save("JTB_reference"+str(i)+str(j)+".png")
        
        return FR,FG,FB,FRrgb,FGrgb,FBrgb
#
#def  JBF(im,sigma_s,sigma_r,r,R,G,B):
#        width,height = im.size
#        FR = np.zeros((width,height), dtype=np.int)
#        ima = Image.new('RGB',(width,height))
#        FG = np.zeros((width,height), dtype=np.int)
#        FB = np.zeros((width,height), dtype=np.int)
#        RR = np.zeros((width+2*r,height+2*r), dtype=np.float)
#        GG = np.zeros((width+2*r,height+2*r), dtype=np.float)
#        BB = np.zeros((width+2*r,height+2*r), dtype=np.float)
#        
#        RR[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = R/255
#        GG[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = G/255
#        BB[r:(np.size(gray_pixel,0)+r),r:(np.size(gray_pixel,1)+r)] = B/255
#        a,b = np.meshgrid(np.arange(-r,+r+1,1),np.arange(-r,+r+1,1))
#        Gs = np.exp(-(a**2+b**2)/(2*sigma_s**2))
#        for y in range(r,height+r):
#                        for x in range(r,width+r):
#                                a,b = np.meshgrid(np.arange(x-r,x+r+1,1),np.arange(y-r,y+r+1,1))
#                                rr = RR[a[:,:],b[:,:]]
#                                gg = GG[a[:,:],b[:,:]]
#                                bb = BB[a[:,:],b[:,:]]
#                                Gr = np.exp(-(((rr-RR[x][y])**2+(gg-GG[x][y])**2+(gg-BB[x][y])**2)/(2*sigma_r**2)))
#                            
#                                FuR = sum(sum(Gr*Gs*rr))
#                                FuG = sum(sum(Gr*Gs*gg))
#                                FuB = sum(sum(Gr*Gs*bb))
#                                Fd = sum(sum(Gr*Gs))
#                                FR[x-r][y-r] = int(FuR/Fd*255)
#                                FG[x-r][y-r] = int(FuG/Fd*255)
#                                FB[x-r][y-r] = int(FuB/Fd*255)
#                                rgba = (FR[x-r][y-r],FG[x-r][y-r],FB[x-r][y-r])
#                                ima.putpixel((x-r,y-r),rgba)
#                                
#        ima.save("JTB_reference.png")
#        return FR,FG,FB




def rgb2gray(im,R,G,B,w_R,w_G,w_B,number): 
        width,height = im.size
        gray = Image.new('L',(width,height))
        gray_pixel = np.zeros((width,height), dtype=np.float)
        for y in range(height):
                for x in range(width):
                        rgba = (w_R*R[x][y]+  # R
                                w_G*G[x][y]+  # G
                                w_B*B[x][y])  # B
                        gray.putpixel((x,y),int(rgba))
                        gray_pixel[x][y]=(rgba)
                       
        gray.save("gray"+str(number) + ".png")
        return gray_pixel/255
    
def rgb(im):
        width,height = im.size
        R = np.zeros((width,height), dtype=np.int)
        G = np.zeros((width,height), dtype=np.int)
        B = np.zeros((width,height), dtype=np.int)
        for y in range(height):
                for x in range(width):
                        rgba = im.getpixel((x,y))
                        R[x][y]=rgba[0]
                        G[x][y]=rgba[1]
                        B[x][y]=rgba[2]
                        im.putpixel((x,y),(R[x][y],G[x][y],B[x][y]))
        return R,G,B


def cost(FR,FG,FB,R,G,B,im):
    width,height = im.size
    total = 0.0
    for x in range(width):
        for y in range(height):
            total += abs(FR[x][y]-R[x][y])+abs(FR[x][y]-R[x][y])+abs(FR[x][y]-R[x][y])
    return total




if __name__ == '__main__':
        im = Image.open("2a.png")
        width,height = im.size
        im.convert('L').save("2a_y.png")
        sigma_s = np.array([1,2,3])
        sigma_r = np.array([0.05,0.1,0.2])
        r = 3*sigma_s  #ws = 2*r+1
        
        R,G,B = rgb(im)
        F = np.zeros((width,height), dtype=np.float)
        total = np.zeros((9,66), dtype=np.float)
        number =0
        p=[]
        u = [0,0,0]
        for  wr in np.arange(0,1.1,0.1):
                for wg in np.arange(1-wr,-0.1,-0.1):
                        wb = 1-wr-wg
                        if wb>=0:
                            gray_pixel = rgb2gray(im,R,G,B,wr,wg,wb,number)
                            n = 0
                            for i in range(3):
                                     for j in range(3):
                                            FR,FG,FB,FR_r,FG_r,FB_r= (joint_bilateral_filter(im,sigma_s[i],sigma_r[j],r[i],gray_pixel,R,G,B))
                                            total[n][number] = cost(FR,FG,FB,FR_r,FG_r,FB_r,im)
                                            n+=1
                            print("frame:"+ str(number))
                            number +=1
        for a in range(9):
            mini= argrelextrema(total[a,:], np.less)
            o = np.asarray([a for b in mini for a in b])
            p = np.hstack((o,p))
        
#        for i in range(3):
#                 for j in range(3):
#                    number = 0
#                    for  wr in np.arange(0,1.1,0.1):
#                            for wg in np.arange(1-wr,-0.1,-0.1):
#                                    wb = 1-wr-wg
#                                    if wb>=0:
#                                        gray_pixel,R,G,B = rgb2gray(im,wr,wg,wb,i,j)
##                                        gray_pixel = rgb2gray(im,0,1,0)/255 #normalize
#                                        F= (joint_bilateral_filter(im,sigma_s[i],sigma_r[j],r[i],gray_pixel,R,G,B))
#                                        total[i+j][number] = cost(F,gray_pixel,im)
#                                        number +=1
#                                        print("frame:"+ str(number))
##        [a for b in argrelextrema(total, np.less) for a in b]
                 
                 
        p = p.astype(int)
        s = np.bincount(p)
        for a in range(3):
            u[a] = np.argmax(s)
            s[u[a]] = 0
        
        ##20,29,37,44,50
        
        #11 21 30
    
        
        
        
        
        
        