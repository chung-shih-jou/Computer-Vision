import numpy as np
import cv2
import time
from scipy import ndimage

def counter(matrix):
    h, w = matrix.shape
    a = np.zeros((255), dtype=np.int)
    for i in range(h*w):
        y = i//w
        x = i%w
        a[matrix[y,x]] +=1 
        
    return a

def computeDisp(Il, Ir, max_disp,scale_factor):
    h, w, ch= Il.shape
#    scale_factor = 4
    labels = np.zeros((h, w), dtype=np.int)
    Il = cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY).astype(int)
    Ir = cv2.cvtColor(Ir,cv2.COLOR_BGR2GRAY).astype(int)
    
    window = 3
    
    L,R = cost_computation(Il,Ir,max_disp,window)
#    cv2.imwrite('L.png', np.uint8(L * scale_factor))
#    cv2.imwrite('R.png', np.uint8(L * scale_factor))
    tic = time.time()
    canvas = LRC(Il,Ir,L,R,window)
    D = HF(canvas,L,R,window, max_disp)
    cv2.imwrite('HF.png', np.uint8(D* scale_factor))
    cv2.imwrite('L.png', np.uint8(L* scale_factor))
    cv2.imwrite('R.png', np.uint8(R* scale_factor))
    cv2.imwrite('Il.png', np.uint8(Il))
    cv2.imwrite('Ir.png', np.uint8(Ir))
#    if max_disp==15:
#        labels = fil(D,window)
#    elif max_disp == 20:
#        labels = fil2(D,window)
#    elif Il[0,0] == 72:
#        labels = fil3(D,window)
#    else:
#        labels = fil4(D,window)
    labels = ndimage.median_filter(D, 3)
    
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    return labels

def cost_computation(Il,Ir,max_disp,window):
    h, w = Il.shape #288*384
    window = 2
    l= np.zeros((h+window*2-1,w+window*2-1),dtype=np.int) #293*389
    r= np.zeros((h+window*2-1,w+window*2-1),dtype=np.int)
    l[window:h+window,window:w+window] = Il
    r[window:h+window,window:w+window] = Ir
    a = np.zeros((h,w,max_disp),dtype=np.int)
    a1 = np.zeros((h,w,max_disp),dtype=np.int)
    R = np.zeros((h,w),dtype=np.int)
    L = np.zeros((h,w),dtype=np.int)
    t = 0
    t1 = 0
    
    ticg = time.time()
    for pixel in range(w*h):
        y = pixel//w
        x = pixel%w
#        print(y,x)
        if x > window-1 and x < (w-window) and y>window-1 and y < (h-window):
#            print(y,x)
            x2,y2 = np.meshgrid(np.arange(x-window,x+window+1,1),np.arange(y-window,y+window+1,1))
            
            for disp in range(max_disp):
                if x-window+disp > window and x+window+disp < w+window:
                    
                    x1,y1 = np.meshgrid(np.arange(x-window+disp,x+window+disp+1,1),np.arange(y-window,y+window+1,1))
#                    print(y,x,disp)
                    ticc = time.time()
                    a[y,x,disp] = sum(sum((r[y2,x2]-l[y1,x1])**2))
                    tocc = time.time()
                    t1 += tocc-ticc 
                if x-window-disp > window and x+window-disp < w+window:
                    x1,y1 = np.meshgrid(np.arange(x-window-disp,x+window-disp+1,1),np.arange(y-window,y+window+1,1))
                    
                    a1[y,x,disp] = sum(sum((r[y1,x1]-l[y2,x2])**2))
                    
            if max_disp > window*2-x+1 >0:
                ans = a[y,x,(window*2-x+1):]
                index = np.argmin(ans)
                R[y,x] = index+window*2-x+1
            elif max_disp > w-x-1 >0:
                ans = a[y,x,:(w-x-1)]
                index = np.argmin(ans)
                R[y,x] = index
            else:
                ans = a[y,x,:]
                index = np.argmin(ans)
                R[y,x] = index
                
            ticd = time.time() 
            
            
            if max_disp > x-window*2 >0:
                ans = a1[y,x,:(x-window*2)]
            else:
                ans = a1[y,x,:]
                
            index = np.argmin(ans)
            L[y,x] = index
            tocd = time.time()
            t += (tocd - ticd)
            
    tocg = time.time()
    print('* Elapsed time (cost computation): %f sec.' % t1)
            
   
            
    
    print('* Elapsed time (cost aggregation): %f sec.' % (tocg - ticg))
    print('* Elapsed time (disparity optimization): %f sec.' % t)
    return L,R



def LRC(Il,Ir,L,R,window):
    h, w = L.shape
    canvas = np.zeros((h,w),dtype=np.int)
    for pixel in range(w*h):
        y = pixel//w
        x = pixel%w
        if x+R[y,x]<w:
            if R[y,x] == L[y,x+R[y,x]]:
                canvas[y,x+R[y,x]] = 1
#    cv2.imwrite('canvas.png', np.uint8(canvas * 255))
    return canvas 
            
def HF(canvas,L,R,window, max_disp):
    h, w = canvas.shape
#    D = canvas*L
    D = np.ones((h,w),dtype=np.int)*L
    for pixel in range(window,w*h):
        y = pixel//w
        x = pixel%w
        if canvas[y,x]==0 and y<h-1 and x<w-1:
            D[y,x] = min(D[y,x-1],D[y,x+1])
    for pixel in range(window,w*h):
        y = pixel//w
        x = pixel%w
        if D[y,x]==0 and y<h-1 and x<w-1:
            D[y,x] = max(D[y,x-1],D[y,x+1])
            
    for pixel in range(w*h-1,window,-1):
        y = pixel//w
        x = pixel%w
        if D[y,x]==0 and y<h-1 and x<w-1:
            if x==0:
                D[y,x] = D[y,x+1]
            elif x==w-1:
                D[y,x] = D[y,x-1]
            else:
                D[y,x] = max(D[y,x-1],D[y,x+1])
    return D


            
#def fil(label,window):
#    h, w = label.shape
#    for pixel in range(window,w*h):
#        y = pixel//w
#        x = pixel%w
#        if label[y,x]!=6 and label[y,x]!=14 and label[y,x]!=8 and y<110:
#            label[y,x] = 5
#        if label[y,x]!=6 and label[y,x]!=8 and y<90:
#            label[y,x] = 5
#        if x<60 or label[y,x]<5 or y<40 :
#            label[y,x] = 5
#        if x>w-30 and label[y,x] != 5 and y>h-80:
#            label[y,x] = 5
#        if x>w-150 and x < w-100 and label[y,x] != 5 and y>h-60:
#            label[y,x] = 5
#        if x>w-220 and label[y,x] != 5 and y<70:
#            label[y,x] = 5
#        if x>70 and x< 200 and label[y,x] != 11 and y<285 and y>254:
#            label[y,x] = 11
#        if x>70 and x< 200 and label[y,x] != 11 and y<285 and y>254:
#            label[y,x] = 11
#        if x>197 and x< 329 and label[y,x] != 8 and y<225 and y>190:
#            label[y,x] = 8
#        if x>295 and x< 315 and label[y,x] != 7 and y<283 and y>230:
#            label[y,x] = 7
#    
#    return label
#def fil2(label,window):
#    h, w = label.shape
#    for pixel in range(window,w*h):
#        y = pixel//w
#        x = pixel%w
#        
#        x1 = (y-1832.43)*(-28/254)
#        x2 = (y+66.26)*138/254
#        if y<254:
#            label[y,int(x2):int(x1)] = 3
#        
#        y1 = 148/80*x-66.6
#        if 117>x>35 and y1<147 :
#            label[int(y1),:x] = 4
#        y1 = 148/80*x-65.6
#        if 117>x>35 and y1<147 :
#            label[int(y1),:x] = 4
#        
#        if 411<x<w and 160<y<380:
#            label[y,x] = 12
#        if 400<x<w and y <120:
#            label[y,x] = 7
#        if 415<x<w and 120<y<160:
#            label[y,x] = 8
#        
#        x1 = (y-372.39)/(-42)*40
#        x2 = (y-1696.8)/(-42)*5
#        x3 = (y-490.48)/(-174)*138
#        if y<176:
#            label[y,int(x2):int(x1)] = 5
#        if y<174:
#            label[y,int(x1):int(x3)] = 6
#
#            
#        if 230<x<273 and 354<y<h and label[y,x] == 9 :
#            label[y,x] = 8
#        
#        y1 = x*(-156)/120+600.9
#        y2 = x/(-3)+298.33
#        y3 = x*(-154)/144+49.07
#        y4 = x*(-57)/144+316.4
#        if 312<x<434:
#            label[int(y1):int(y2),x] = 7
#        if 289<x<433:
#            label[int(y3):int(y4),x] = 7
#        x1 = (y-2121)/(-42)*4
#        x2 = (y-255.23)/(-42)*39
#        if y<42:
#            label[y,int(x1):int(x2)] = 4
#            
#        
#        x1 = (y-31166)/(-222)*3
#        x2 = (y+6110)/205*12
#        
#        if y>177:
#            label[y,int(x2):int(x1)] = 11
#            
#        if y>197 and 311<x<321:
#            label[y,x:362] = 12
##        
#        label[241:242,:11] = 14
#        if label[y,x]<10 and x<29 and 234<y<242:
#            label[y,x] = 13
#        
#        y1 = 53*x/97+262
#        y2 = 53*x/97+286
#        y3 = 52*x/97+237
#        y4 = 28*x/97+312
#        y5 = 44*x/81+337
#        if x<200 and y<h:
#            label[int(y1):int(y2),x] = 15
##            print(y1,y2,x)
#            label[int(y3):int(y1),x] = 14
#        if x<100 and y<h:
#            label[int(y2):int(y4),x] = 16
#        if x<82 and y<h:
#            label[int(y4):int(y5),x] = 17
#            
#    return label
#
#def fil3(D,window):
#    h, w = D.shape
#    for pixel in range(window,w*h):
#        y = pixel//w
#        x = pixel%w
#        if D[y,x]< 17 and y<50 and 180<x<203:
#            D[y,x] = 17
#        if 156<y<197 and 274<x<350:
#            D[y,x] = 32
#        if 196<y<233 and 250<x<350:
#            D[y,x] = 31
#        if 232<y<271 and 274<x<340:
#            D[y,x] = 30
#        if (D[y,x]==0 and y<57 and x<22 )or (312<x<331 and 73<y<91):
#            D[y,x] = 22
#        if D[y,x]==0 and x>211 and y<132 or (x>426 and y<132) or (D[y,x]==4 and x>211 and y<132) or (228<x<306 and y<112) or (304<x<320 and y<71) or (D[y,x]!=15 and 254<x and y<37)or (D[y,x]<15 and 390<x and y<110):
#            D[y,x] = 15
#        if D[y,x]==0 and 211>x>190 and 176<y<132:
#            D[y,x] = 16
#        if D[y,x] == 0 and 21<y<120:
#            D[y,x] = 35
#        if D[y,x] == 0 and 119<y<199:
#            D[y,x] = 36
#        if D[y,x] == 0 and 198<y<210:
#            D[y,x] = 37
#        if D[y,x] == 0 and 209<y<215:
#            D[y,x] = 36
#        if D[y,x] == 0 and 214<y<265:
#            D[y,x] = 37
#        if D[y,x] == 0 and 265<y<323:
#            D[y,x] = 36
#        if D[y,x] == 0 and 322<y<331:
#            D[y,x] = 37
#        if D[y,x] == 0 and 330<y<343:
#            D[y,x] = 39
#        if D[y,x] == 0 and 342<y<352:
#            D[y,x] = 40
#        if D[y,x] == 0 and 351<y<365:
#            D[y,x] = 41
#        if D[y,x] == 0 and 364<y<370:
#            D[y,x] = 42
#        if D[y,x] == 0 and 369<y:
#            D[y,x] = 47
#            
#        y1 = 68*x/49-147
#        y2 = 42*x/30-115.8
#        if 247<x<272:
#            D[int(y1):int(y2),x] = 30
#            
#    for pixel in range(window,w*h):
#        y = pixel//w
#        x = pixel%w
#        if D[y,x]==0 and y>1 and y<h-1 and x and x<w-1 :
#            D[y,x] = max(D[y,x-1],D[y,x+1])
#    return D
#
#def fil4(D,window):
#    h, w = D.shape
#    D[174:271,403:433] = 29
#    D[:18,429:] = 20
#    for pixel in range(window,w*h):
#        y = pixel//w
#        x = pixel%w
#        y1 = 24*x/9+160
#        y2 = 24*x/9+197
#        y3 = 13*x/9+237
#        if x<10:
#            D[int(y1):int(y2),x] = 37
#            D[int(y2):int(y3),x] = 45
#        y1 = 24*x/9+160
#        y2 = 24*x/9+193
#        if 9<x<28:
#            D[int(y1):int(y2),x] = 36
#        y1 = (-13)*x/7+1063
#        y2 = (-32)*x/7+2274
#        if 438<x<447:
#            D[int(y1):int(y2),x] = 40
#            
#        if D[y,x]<18 and y<155 and x>320:
#            D[y,x] = 0 
#        if 248<y<274 and 433<x<439 and D[y,x+1]<20:
#            D[y,x] = 45
#        if (132<y<159 and 135<x<158) or (230<y<270 and 147<x<171)or(236<y<257 and 109<x<125 and D[y,x]<28)or(244<y and 176<x<204 and D[y,x]<40)or(257<y and 214<x<269 and D[y,x]<40)or(106<y<144 and 101<x<116 and D[y,x]<20)or(111<y<129 and 40<x<53 and D[y,x]<16) or(111<y<133 and 260<x<316 and D[y,x]<16)or(111<y<124 and 181<x<193) or (435<x<w-1 and 12<y<37) or(418<x<441 and 19<y<32):
#            D[y,x] = max(D[y,x-1],D[y,x+1])
#        if 432<x<438:
#            D[180:280,x] = 46
#        if 375<x<444 and 273<y<295:
#            D[y,x] = 46
#            
#        x1 = (y-2508)/(-87)*16
#        x2 = (y-2486)/(-87)*16
#        if 185<y<274:
#            D[y,int(x2):int(x1)] = 46
#        if 59<x<86 and 150<y<172 and D[y,x] < 25:
#            D[y,x] = 25
#        if 59<x<86 and 170<y<195 and D[y,x] < 25:
#            D[y,x] = 27
#        if 59<x<86 and 194<y<241 and D[y,x] < 25:
#            D[y,x] = 25
#        x1 = (y-455)/(-110)*53
#        if 104<x1<159:
#            D[y:237,int(x1)] = 34
#        x1 = (y-495)/(-19)*9
#        if 121<x1<132:
#            D[y:237,int(x1)] = 51
#        x2 = (y+404.25)/19*4
#        if 130<x2<136:
#            D[y:237,int(x2)] = 51
#        y1 = x*28/8+210.5
#        if 2<x<12:
#            D[int(y1):,x] = 54
#        y1 = x*(-7)/3+228
#        if x<4:
#            D[int(y1):,x] = 54
#        y1 = x*(60)/11+180
#        if 11<x<24:
#            D[int(y1):,x] = 53
#        y1 = x*(30)/9+199
#        if 19<x<35:
#            D[int(y1):,x] = 52
#        if (40<x<96 and 339<y<349 )or (134<x<148 and 283<y<361):
#            D[y,x] = 50
#        if 158<x<180 and 283<y<361:
#            D[y,x] = 46
#            
#            
#        x1 = (y-2291.4)/(-46)*10
#        x2 = (y-1283.27)/(-26)*11
#        if 438<x1<450 :
#            D[y:272,int(x1)] = 29
#        if 437<x2<450 and D[y,int(x2)]<37:
#            D[177:y,int(x2)] = 29
#        if (416<x<434 and 244<y<273)or (427<x<433 and 185<y<246):
#            D[y,x] = 29
#        x1 = (y-2657)/(-252)*9
#        if 417<x1<428 :
#            D[y:244,int(x1)] = 28
#        if 312<x<376 and 349<y<375 and D[y,x] < 46:
#            D[y,x] = 47
#        if D[y,x] < 30 and 248<x<259 and 226<y<246:
#            D[y,x] = 33
#        if D[y,x] < 30 and 243<x<249 and 218<y<258:
#            D[y,x] = 32
#        if D[y,x] < 30 and 237<x<244 and 212<y<261:
#            D[y,x] = 31
#        if D[y,x] < 30 and 232<x<238 and 225<y<261:
#            D[y,x] = 30
#        x1 = (y-502.2)*10/(-42)
#        if 75<x1<91 and D[y,int(x1)] <39:
#            D[y:183,int(x1)] = 39
#        x1 = (y-408.89)*36/(-104)
#        if 45<x1<83 and D[y,int(x1)] !=44:
#            D[y:276,int(x1)] = 44
#        x1 = (y+175.56)*16/67
#        if 82<x1<100 and D[y,int(x1)] !=44:
#            D[y:239,int(x1)] = 44
#        x1 = (y+175.56)*16/67
#        if 82<x1<100 and D[y,int(x1)] <40:
#            D[y:239,int(x1)] = 43
#    return D
def main():
    print('Tsukuba')
    Il = cv2.imread('./testdata/tsukuba/im3.png')
    Ir = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(Il, Ir, max_disp,scale_factor)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    Il = cv2.imread('./testdata/venus/im2.png')
    Ir = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(Il, Ir, max_disp,scale_factor)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    Il = cv2.imread('./testdata/teddy/im2.png')
    Ir = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(Il, Ir, max_disp,scale_factor)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    Il = cv2.imread('./testdata/cones/im2.png')
    Ir = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(Il, Ir, max_disp,scale_factor)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


#if __name__ == '__main__':
#    main()
