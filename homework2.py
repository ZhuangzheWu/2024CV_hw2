# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt
# import utils
import random
import statistics
import math
IMGDIR = 'Problem2Images'


def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we pply?
    # which kernel should we choose to calculate gradient_x?
    # TODO
    # img=ndimage.gaussian_filter(img,0.5)
    
    grad_x=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # grad_x=ndimage.sobel(img,axis=0)
    return grad_x

def gradient_y(img):
    # TODO
    # img=ndimage.gaussian_filter(img,0.5)
    grad_y=cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return grad_y

def harris_response(img, alpha, win_size):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 29 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO   
    Ix = gradient_x(img) 
    Iy = gradient_y(img)   
    # Ix=cv2.GaussianBlur(Ix,(3,3),0.5)
    # Iy=cv2.GaussianBlur(Iy,(3,3),0.5)
    Ix2 = Ix ** 2  
    Iy2 = Iy ** 2
    Ixy = Ix * Iy  # 计算Ix和Iy的乘积
 
    Sx2 = cv2.GaussianBlur(Ix2, (win_size, win_size), 1)
    Sy2 = cv2.GaussianBlur(Iy2, (win_size, win_size), 1)
    Sxy = cv2.GaussianBlur(Ixy, (win_size, win_size), 1)
    
    # 计算 Harris 响应 R
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    R = det_M - alpha * (trace_M ** 2)  
    return R



def corner_selection(R, thresh, min_dist):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    # TODO
    
    R_max=ndimage.maximum_filter(R,size=min_dist)
    R_local_max = (R == R_max)
    
    R_selection = np.zeros_like(R)
    R_selection[(R_local_max) & (R >= thresh)] = R[(R_local_max) & (R >= thresh)]

    indices = np.argwhere(R_selection > thresh)
    pix=[tuple(idx) for idx in indices]
    return pix

def histogram_of_gradients(img, pix):
    # no template for coding, please implement by yourself.
    # You can refer to implementations on Github or other weblock_sizeites
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose n*n blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again. 
    # TODO
    grad_x=gradient_x(img)
    grad_y=gradient_y(img)

    mags=np.sqrt(grad_x**2+grad_y**2)
    angles=np.arctan2(grad_y,grad_x)*(180/np.pi)
    # # angles[angles<0]+=180
    # angles
    # bin_width=360/8
    features=[]
    block_size=2
    cell_size=8
    bins=8
    for kp in pix:      
        x,y=kp
    #     x_start=max(0,x-8)
    #     y_start=max(0,y-8)
    #     x_end=min(img.shape[0],x+8)
    #     y_end=min(img.shape[1],y+8)
    #     block_mag = mags[x_start:x_end,y_start:y_end]
    #     block_angle = angles[x_start:x_end,y_start:y_end]
        hist=np.zeros((block_size*block_size,bins))
        n=0
        for i in range(-block_size//2,block_size//2):
            for j in range(-block_size//2,block_size//2):
                cx=x+i*cell_size
                cy=y+j*cell_size
                if 0<=cx<img.shape[0] and 0<=cy<img.shape[1]:
                    for k in range(cell_size):
                        for l in range(cell_size):
                            if 0<=cx+k<img.shape[0] and 0<=cy+l<img.shape[1]:
                                bin_index1=int((angles[cx+k,cy+l]/360)*bins)%bins
                                hist[n][bin_index1]+=mags[cx+k,cy+l]
                n+=1

        est=np.argmax(hist)%bins
        first=hist[:,est:bins]
        second=hist[:,0:est]
        histogram=np.hstack((first,second))
        histogram=np.reshape(histogram,(block_size*block_size*bins,))
        histogram/=np.linalg.norm(histogram)+1e-6
        histogram=np.roll(histogram,-np.argmax(histogram))
        features.append(histogram)
    
    
    return features

def feature_matching(img_1, img_2):
    if len(img_1.shape) == 3:  # 如果是彩色图像（3个通道）
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    
    if len(img_2.shape) == 3:  # 如果是彩色图像（3个通道）
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    R1 = harris_response(img_1, 0.04, 5)
    R2 = harris_response(img_2, 0.04, 5)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 3)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 3)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    threshold = 0.6
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    # hist_slope=np.zeros(10)  # 10个角度的直方图
    bin_width=180/10
    hist_length=[]
    angles=[]
    
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p]) <= threshold:
                x1, y1 = cor1[p]
                x2, y2 = cor2[pos]
                slope = (x2 - x1) / (y2 - y1+1e-2)
                length=np.sqrt((x2-x1)**2+(y2-y1)**2)
                hist_length.append(length)
                angle=np.arctan(slope)*180/np.pi
                if angle<0:
                    angle+=180
                bin_index = int(angle // bin_width)
                angles.append(bin_index)
                # hist_slope[bin_index]+=1
                # # hist_slope[bin_index][1]=
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                x1, y1 = cor1[pos]
                x2, y2 = cor2[p]
                slope = (x2 - x1) / (y2 - y1+1e-2)
                length=np.sqrt((x2-x1)**2+(y2-y1)**2)
                hist_length.append(length)
                angle=np.arctan(slope)*180/np.pi
                if angle<0:
                    angle+=180
                bin_index = int(angle // bin_width)
                angles.append(bin_index)
                # hist_slope[bin_index]+=1
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    # i=0
    # angle_index=hist_slope.argmax()
    # mode=statisticell_size.mode(hist_length)        
    # i=0  
    # idx=0
    # while i <np.shape(pixels_1)[0]:
    #     if(ablock_size(hist_length[idx]-mode)>10):
    #     # if(angles[idx]!=angle_index and ablock_size(hist_length[idx]-mode)>20):
    #         pixels_1.pop(i)
    #         pixels_2.pop(i)
    #     else:
    #         i+=1
    #     idx+=1
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])
    rate = np.shape(pixels_1)[0]/min_len
    assert np.shape(pixels_1)[0]>=4, "Fail to Match!"#0.03 
    return pixels_1, pixels_2

def sift_matching(img_1, img_2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    pixels_1 = [kp1[m.queryIdx].pt for m in good]
    pixels_1 = [(y, x) for x, y in pixels_1]
    pixels_2 = [kp2[m.trainIdx].pt for m in good]
    pixels_2 = [(y, x) for x, y in pixels_2]
    return pixels_1, pixels_2

def test_matching(img_1,img_2):    
    
    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_gray_1, img_gray_2)
    # pixels_1, pixels_2 = sift_matching(img_gray_1, img_gray_2)
    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        plt.plot([y1, y2+W_1], [x1, x2])

    # plt.show()
    plt.savefig('test.jpg')

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    len = np.shape(pixels_1)[0]
    A = np.zeros((2*len, 9))
    for i in range(len):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        A[2*i, :] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        A[2*i+1, :] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]
    U, S, V = np.linalg.svd(A)
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    homo_matrix = homo_matrix/homo_matrix[2, 2]
    return homo_matrix
idx=0
def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    # TODO
    iterations = 300000
    global idx
    threshold = 3
    best_homo = None
    max_inliers = 0
    num_points = len(pixels_1)
    # iterations=int(math.comb(num_points,4)*0.03)+10000
    pixels_1=np.array(pixels_1)
    pixels_2=np.array(pixels_2)
    for _ in range(iterations):
        # 随机选择4个点
        sample_indices = random.sample(range(num_points), 4)
        src_sample = pixels_1[sample_indices]
        dst_sample = pixels_2[sample_indices]
        
        # 计算当前样本的单应性矩阵
        homo_matrix = compute_homography(src_sample, dst_sample)

        # 计算当前单应性矩阵下的所有点的投影
        src_homo = np.hstack((pixels_1, np.ones((len(pixels_1), 1))))
        projected_points = (homo_matrix @ src_homo.T).T
        projected_points /= projected_points[:, 2].reshape(-1, 1)  # 归一化
        
        # 计算内点数量
        distances = np.sqrt(np.sum((projected_points[:, :2] - pixels_2) ** 2, axis=1))
        inliers = distances < threshold

        if inliers.sum() > max_inliers:
            max_inliers = inliers.sum()
            best_homo = homo_matrix

    # best_homo = compute_homography(pixels_1, pixels_2)
    idx+=1
    return best_homo

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)
    p_test=est_homo.dot(np.array([248, 178, 1]))
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, w1, 1]))
    p3 = est_homo.dot(np.array([h1, 0, 1]))
    p4 = est_homo.dot(np.array([h1, w1, 1]))
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    p_test = np.int16(p_test/p_test[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(h2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(w2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x).transpose()
    y = np.float32(y).transpose()
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    trans_x = trans_x/(trans_z+1e-4)
    trans_y = trans_y/(trans_z+1e-4)
    est_img_1 = cv2.remap(img_1, trans_y, trans_x, cv2.INTER_LINEAR)
    # plt.imshow(est_img_1)
    # plt.show()
    est_img_2 = cv2.remap(img_2, y, x, cv2.INTER_LINEAR)
    # plt.imshow(est_img_2)
    # plt.show()
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_y,
                       trans_x, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), y, x, cv2.INTER_LINEAR)
    
    mask_img_1 = cv2.inRange(est_img_1, (1, 1, 1), (255, 255, 255))  # 定义阈值范围
    mask_img_2 = cv2.inRange(est_img_2, (1, 1, 1), (255, 255, 255))
    mask_img_1 = cv2.merge([mask_img_1, mask_img_1, mask_img_1])  # 使掩码适配三通道
    mask_img_2 = cv2.merge([mask_img_2, mask_img_2, mask_img_2])     
    alpha1 = alpha1 * (mask_img_1.astype(np.float32) / 255)  
    alpha2 = alpha2 * (mask_img_2.astype(np.float32) / 255)  
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1 + est_img_2*alpha2
    # est_img=cv2.rotate(est_img, cv2.ROTATE_90_CLOCKWISE)
    return est_img


def generate_panorama(ordered_img_seq):
    # len = np.shape(ordered_img_seq)[0]
    l=len(ordered_img_seq)
    mid = int(l/2) # middle anchor
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < l):
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        # pixels1, pixels2 = sift_matching(ordered_img_seq[j], principle_img)
        test_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        # pixels1, pixels2 = sift_matching(ordered_img_seq[i], principle_img)
        test_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano

if __name__ == '__main__':
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements
    # test_matching()
    
    # an example
    # img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn10.jpg')
    # img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn11.jpg')
    # img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn12.jpg')
    # img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn13.jpg')
    # img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn14.jpg')
    # img_6 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn15.jpg')
    # img_7 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn16.jpg')
    # img_1=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0175.jpg')
    # img_2=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0176.jpg')
    img_3=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0184.jpg')
    img_4=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0185.jpg')
    img_5=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0186.jpg')
    # img_6=cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0187.jpg')
    # img_1=cv2.imread(f'{IMGDIR}/panoramas/grail/grail01.jpg')
    # img_2=cv2.imread(f'{IMGDIR}/panoramas/grail/grail02.jpg')
    # img_3=cv2.imread(f'{IMGDIR}/panoramas/grail/grail03.jpg')
    # img_4=cv2.imread(f'{IMGDIR}/panoramas/grail/grail04.jpg')
    # img_5=cv2.imread(f'{IMGDIR}/panoramas/grail/grail05.jpg')
    # img_6=cv2.imread(f'{IMGDIR}/panoramas/grail/grail06.jpg')
    # img_1=cv2.imread(f'{IMGDIR}/2_1.jpg')
    # img_2=cv2.imread(f'{IMGDIR}/2_2.jpg')
    # img_1=cv2.imread(f'{IMGDIR}/panoramas/library/1.jpg')
    # img_2=cv2.imread(f'{IMGDIR}/panoramas/library/2.jpg')
    # img_3=cv2.imread(f'{IMGDIR}/panoramas/library/3.jpg')
    # img_4=cv2.imread(f'{IMGDIR}/panoramas/library/4.jpg')
    # img_5=cv2.imread(f'{IMGDIR}/panoramas/library/5.jpg')
    # img_6=cv2.imread(f'{IMGDIR}/panoramas/library/6.jpg')
       
    img_list=[]
    
    # img_list.append(img_1)
    # img_list.append(img_2)
    img_list.append(img_3)
    
    img_list.append(img_4)
    img_list.append(img_5)
    # img_list.append(img_6)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/test.jpg", pano)
