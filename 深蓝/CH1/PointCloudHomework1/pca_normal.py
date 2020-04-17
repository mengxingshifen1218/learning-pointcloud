# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
import numpy as np
import open3d
import time

def PCA(data, correlation=False, sort=True):

    # 去中心化
    data_mean = data - np.mean(data,axis=0) # n*3

    # 计算协方差矩阵
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(data_mean.T,data_mean)) # (3,n)*(n,3)

    # 从大到小排序
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def main():

    # 读取点云数据并显示
    init_points = np.loadtxt('./data/airplane_0001.txt', delimiter=',').astype(np.float32)[:, 0:3]
    pointcloud = open3d.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(init_points)
    # open3d.visualization.draw_geometries([pointcloud],window_name='init_pointcloud')

    # 用PCA分析点云主方向
    w, v = PCA(init_points)
    # print(v[:, 0:2])

    point_cloud_vector = v[:, 0]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)

    pca_v = np.array([[0, 0, 0], v[:, 0], v[:, 1], v[:, 2]], dtype=np.float32)
    pca_lines = [[0, 1], [0, 2],[0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0 , 0, 1]]  # RGB

    # 定义PCA的三条线
    pca_line = open3d.LineSet()
    pca_line.lines = open3d.Vector2iVector(pca_lines)
    pca_line.colors = open3d.Vector3dVector(colors)
    pca_line.points = open3d.Vector3dVector(pca_v)

    # 显示PCA
    # open3d.visualization.draw_geometries([pointcloud, pca_line],window_name='show_pca',height=500,width=600)

    # 显示投影点云
    # pca_v = np.hstack((v[:, 0:2], np.zeros((3,1))))
    # pca_points = np.dot(init_points, pca_v)
    # pointcloud.points = open3d.utility.Vector3dVector(pca_points)
    # open3d.visualization.draw_geometries([pointcloud, pca_line],window_name='show_pca_2',height=500,width=600)

    # 循环计算每个点的法向量
    pointcloud_tree = open3d.geometry.KDTreeFlann(pointcloud) # 创建一个KDTree对象
    points = np.array(pointcloud.points)
    normals = []

    search_radius = 0.1 # 领域半径
    search_num = 100  # 球域内最大点的数量
    start = time.time()

    for i in range(points.shape[0]):
        _, index , _ = pointcloud_tree.search_hybrid_vector_3d(points[i,:], search_radius, search_num)
        set = points[index,:]
        _, set_v = PCA(set)
        normals.append(set_v[:, 2])

    print('use time:',time.time()-start)
    normals = np.array(normals, dtype=np.float64)

    pointcloud.normals = open3d.utility.Vector3dVector(normals)
    open3d.visualization.draw_geometries([pointcloud],window_name='show_normals',height=500,width=600)

if __name__ == '__main__':
    main()


