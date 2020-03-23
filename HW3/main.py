import numpy as np
import cv2



# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N/2 < 4:
        print('At least 4 points should be given')
    A = np.array([[u[0],u[1],1,0,0,0,-u[0]*v[0],-u[1]*v[0],-v[0]],
                      [0,0,0,u[0],u[1],1,-u[0]*v[1],-u[1]*v[1],-v[1]],
                      [u[2],u[3],1,0,0,0,-u[2]*v[2],-u[3]*v[2],-v[2]],
                      [0,0,0,u[2],u[3],1,-u[2]*v[3],-u[3]*v[3],-v[3]],
                      [u[4],u[5],1,0,0,0,-u[4]*v[4],-u[5]*v[4],-v[4]],
                      [0,0,0,u[4],u[5],1,-u[4]*v[5],-u[5]*v[5],-v[5]],
                      [u[6],u[7],1,0,0,0,-u[6]*v[6],-u[7]*v[6],-v[6]],
                      [0,0,0,u[6],u[7],1,-u[6]*v[7],-u[7]*v[7],-v[7]]]).astype(float)
    
    a = np.dot(A.T, A)
    eigValue, eigVect= np.linalg.eig(a)
    value,index= find_nearest( eigValue, 0 )
    H = np.reshape(eigVect[:,index],(3,3))
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    height, width, ch = img.shape
    height = height-1
    width = width-1
    u = np.array([0,0,width,0,0,height,width,height])
    v = np.array([corners[0,0],corners[0,1],corners[1,0],corners[1,1],
                      corners[2,0],corners[2,1],corners[3,0],corners[3,1]])
    # TODO: some magic
    H = solve_homography(u, v)
    
    height = height+1
    width = width+1
    ux,uy = np.meshgrid(np.arange(0,width,1),np.arange(0,height,1))
    a = np.ones(width*height)  ### 1
    ux = np.reshape(ux,(1,width*height)) ###ux
    uy = np.reshape(uy,(1,width*height)) ###uy
    ux = np.r_[ux,uy] 
    ux = np.c_[ux.T,a] 
    c = np.dot(np.reshape(H,(3,3)),ux.T)
    a = np.array([c[2,:],c[2,:],c[2,:]])
    b = np.floor(c[:,:]/a) 
    b = b.astype(int)
    vx = b[1,:]
    vy = b[0,:]
       
    return vx,vy
    
def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])#358 681   436 717  285 856  362 892
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

    # TODO: some magic
    for i in range(5):
        print ('frame:' + str(i+1))
        if i ==0:
            img = img1
            corners = corners1
        elif i ==1:
            img = img2
            corners = corners2
        elif i ==2:
            img = img3
            corners = corners3
        elif i ==3:
            img = img4
            corners = corners4
        else:
            img = img5
            corners = corners5
        
        height = np.size(img,0)-1
        width = np.size(img,1)-1
        vx,vy = transform(img, canvas, corners) 
        
        width = width+1
        height = height+1
        d = np.reshape(img,(width*height,3))
        canvas[vx,vy,:] = d[:,:] 
        
    cv2.imwrite('part1.png', canvas)
    
    
    
    # Part 2
    img = cv2.imread('./input/screen.jpg')
    width = 230
    height = 230
    corners = np.array([[1040,370],[1102,395],[983,554],[1036,601]])
    canvas = np.zeros((height,width,3),dtype=np.float)
    
    vx,vy = transform(canvas,img, corners) 
    
    canvas = np.zeros((width*height,3),dtype=np.float)
    canvas[:,:] = img[vx,vy,:]
    filter1 = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    canvas1 = np.zeros(((width+2),(height+2),3),dtype=np.float)
    output = np.zeros((width,height,3),dtype=np.float)
    canvas = np.reshape(canvas,(230,230,3))
    canvas1[1:231,1:231,:] = canvas[:,:,:]
    for i in range(1,231,1):
        for j in range(1,231,1):
            output[i-1,j-1,0] = sum(sum(canvas1[i-1:i+2,j-1:j+2,0]*filter1))
            output[i-1,j-1,1] = sum(sum(canvas1[i-1:i+2,j-1:j+2,1]*filter1))
            output[i-1,j-1,2] = sum(sum(canvas1[i-1:i+2,j-1:j+2,2]*filter1))
    cv2.imwrite('part2.png', output)



    # Part 3
    img = cv2.imread('./input/crosswalk_front.jpg')
    img1 = np.zeros(img.shape)
    
    # top_top
    x1 = 145
    x2 = 806
    y1 = 147
    y2 = 400
    
    height = y2-y1 #400-147
    width = x2-x1 #806-145
    
    # front_down
#    height = 316-231
#    width = 666-58
    
    corners1 = np.array([[135,161],[584,157],[65,238],[661,231]]) # front_top
#    corners2 = np.array([[145,400],[806,401],[215,567],[737,569]]) # top_down
    corners = corners1
#    top
#    top_top = np.array([[147,147],[802,144],[145,400],[806,401]]) #上
#    top_down = np.array([[145,400],[806,401],[215,567],[737,569]]) #下
#    front
#    front_top = np.array([[135,161],[584,157],[65,238],[661,231]])
#    front_down = np.array([[65,238],[661,231],[58,316],[666,310]])
    
    
    canvas = np.zeros((x1,y1,3),dtype=np.float) #u
    height, width, ch = canvas.shape
    height = height-1
    width = width-1
    u = np.array([x1,y1,x2,y1,x1,y2,x2,y2])
    v = np.array([corners[0,0],corners[0,1],corners[1,0],corners[1,1],
                      corners[2,0],corners[2,1],corners[3,0],corners[3,1]])
    # TODO: some magic
    H = solve_homography(u, v)
    
    height = height+1 ##252
    width = width+1 ##660
    ux,uy = np.meshgrid(np.arange(0,x2,1),np.arange(0,y2+133,1))
    a = np.ones((x2)*(y2+133))  ### 1
    ux = np.reshape(ux,(1,(x2)*(y2+133))) ###ux
    uy = np.reshape(uy,(1,(x2)*(y2+133))) ###uy
    ux = np.r_[ux,uy] 
    ux = np.c_[ux.T,a] 
    c = np.dot(np.reshape(H,(3,3)),ux.T)
    a = np.array([c[2,:],c[2,:],c[2,:]])
    b = np.floor(c[:,:]/a) 
    b = b.astype(int)
    vx = b[1,:]
    vy = b[0,:]
    canvas2 = np.zeros(((y2+133),(x2),3),dtype=np.float)
    d = np.reshape(canvas2,((x2)*(y2+133),3))
    d[:,:] = img[vx,vy,:] 
    d = np.reshape(d,((y2+133),(x2),3))
    cv2.imwrite('part3.png', d)
    
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx

if __name__ == '__main__':
    main()
