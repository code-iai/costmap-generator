import sys
from os.path import isdir
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy.stats import multivariate_normal

GRID_SIZE = 4
GRID_FACTOR = 1000


def mirror_pose_data(data):
    data_x_raw = list(data.t_x)
    data_y_raw = list(data.t_y)
    data_x = []
    data_y = []

    data_x.extend(2 * data_x_raw)
    data_x.extend(2 * [-x for x in data_x_raw])
    data_y.extend(data_y_raw)
    data_y.extend(2 * [-x for x in data_y_raw])
    data_y.extend(data_y_raw)

    return data_x, data_y


def generate_costmaps_for_environment(path_to_environment_csv, object_column, mirror_data=False):
    data = pd.read_csv(path_to_environment_csv)

    for obj in data[object_column].unique():
        print(obj)
        obj_data = data[data[object_column] == obj]
        print(obj_data.success.value_counts())
        success_data = obj_data[obj_data.success == True]
        #success_data = success_data.loc[success_data['t_y'] <= 1.5]
        failed_data = obj_data[obj_data.success == False]
        print(success_data.bodyPartsUsed.value_counts())

        fig = plt.figure()
        if not success_data.empty:
            success_data_x, success_data_y = mirror_pose_data(success_data) if mirror_data else list(
                success_data.t_x), list(success_data.t_y)
            plt.scatter(success_data_x, success_data_y, color='g')
        if not failed_data.empty:
            failed_data_x, failed_data_y = mirror_pose_data(failed_data) if mirror_data else list(
                failed_data.t_x), list(failed_data.t_y)
            plt.scatter(failed_data_x, failed_data_y, color='r')

        plt.title('{}'.format(obj))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        x = np.linspace(-GRID_SIZE, GRID_SIZE, GRID_FACTOR)
        y = np.linspace(-GRID_SIZE, GRID_SIZE, GRID_FACTOR)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        mu = np.array([success_data.t_x.mean(), success_data.t_y.mean()])
        cov = np.cov([success_data.t_x, success_data.t_y])
        rv = multivariate_normal(mu, cov)
        Z = rv.pdf(pos)
        print('Max Success:{}, {}'.format(success_data.t_x.mean(), success_data.t_y.mean()))
        print('MEAN:{}'.format(mu))
        print('VARIANCE:{}'.format(cov))

        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        #ax.contour(X, Y, Z, rstride=5, cstride=5,cmap='viridis', antialiased=True)
        ax.contour(X, Y, Z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.set_zlabel('z')
        plt.title('{}'.format(obj))
        fig.show()


if __name__ == "__main__":

    #generate_costmaps_for_environment('result/environment_bowl.csv', 'information')
    generate_costmaps_for_environment('result/dish_2.csv', 'type')
    #generate_costmaps_for_environment('result/grasping_unreal.csv', 'object_acted_on')
    #generate_costmaps_for_environment('result/placing_bowl.csv', 'object_acted_on')
    # # args = sys.argv[1:]
    # # path = args[0]
    # # result_dir_path = args[1]
    # #
    # # if isdir(path):
    #
    # grasping_data = pd.read_csv('result/environment.csv')
    # for obj in grasping_data.information.unique():
    #     print(obj)
    #     obj_data = grasping_data[grasping_data.information == obj]
    #     print(obj_data.success.value_counts())
    #     success_data = obj_data[obj_data.success == True]
    #     failed_data = obj_data[obj_data.success == False]
    #     print(success_data.bodyPartsUsed.value_counts())
    #     fig = plt.figure()
    #     success_data_x_raw = list(success_data.t_x)
    #     success_data_y_raw = list(success_data.t_y)
    #     success_data_x = []
    #     success_data_y = []
    #
    #     success_data_x.extend(1 * success_data_x_raw)
    #     #success_data_x.extend(2 * [-x for x in success_data_x_raw])
    #     #success_data_y.extend(success_data_y_raw)
    #     #success_data_y.extend(2 * [-x for x in success_data_y_raw])
    #     success_data_y.extend(success_data_y_raw)
    #
    #     failed_data_x_raw = list(failed_data.t_x)
    #     failed_data_y_raw = list(failed_data.t_y)
    #     failed_data_x = []
    #     failed_data_y = []
    #
    #     failed_data_x.extend(1 * failed_data_x_raw)
    #     #failed_data_x.extend(2 * [-x for x in failed_data_x_raw])
    #     #failed_data_y.extend(failed_data_y_raw)
    #     #failed_data_y.extend(2 * [-x for x in failed_data_y_raw])
    #     failed_data_y.extend(failed_data_y_raw)
    #
    #     plt.scatter(success_data_x, success_data_y, color='g')
    #     #plt.scatter(failed_data_x, failed_data_y, color='r')
    #     plt.title('{}'.format(obj))
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()
    #
    #     # training = obj_data[['t_x','t_y']].values.tolist()
    #     # # temp_1 = [[x,-y]for x,y in training]
    #     # # temp_2 = [[-x, y] for x, y in training]
    #     # # temp_3 = [[-x, -y] for x, y in training]
    #     # # training.extend(temp_1)
    #     # # training.extend(temp_2)
    #     # # training.extend(temp_3)
    #     # labels = obj_data.success.values.tolist()
    #     # clf = GaussianNB()
    #     # clf.fit(training, labels)
    #     # x_values = np.linspace(-4,4,1000)
    #     # y_values = np.linspace(-4,4,1000)
    #     # X,Y = np.meshgrid(x_values,y_values)
    #     # Z = []
    #     # for i in range(0,len(X)):
    #     #     x = X[i]
    #     #     y = Y[i]
    #     #     zipped = list(zip(x,y))
    #     #     z_values = [result[1] for result in clf.predict_proba(zipped)]
    #     #     Z.append(z_values)
    #     #
    #     # fig = plt.figure()
    #     # ax = plt.axes(projection="3d")
    #     # ax.plot_surface(X, Y, np.array(Z), cmap='inferno', edgecolor='none')
    #     # ax.set_xlabel('x')
    #     # ax.set_ylabel('y')
    #     # ax.set_zlabel('z')
    #     # plt.title('{}'.format(obj))
    #     # plt.show()
    #
    #     x = np.linspace(-1.5,1.5,100)
    #     y = np.linspace(-1.5,1.5,100)
    #     X, Y = np.meshgrid(x, y)
    #     pos = np.dstack((X, Y))
    #     mu = np.array([success_data.t_x.mean(), success_data.t_y.mean()])
    #     mu = np.array([success_data.t_x.mean(), success_data.t_y.mean()])
    #     cov = np.cov([success_data.t_x, success_data.t_y])
    #     rv = multivariate_normal(mu, cov)
    #     Z = rv.pdf(pos)
    #     print('Max Success:{}, {}'.format(success_data.t_x.mean(), success_data.t_y.mean()))
    #     plot_results.append([X, Y, Z])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(*plot_results[0], cmap='winter', antialiased=True)
    # ax.plot_surface(*plot_results[1], cmap='inferno', antialiased=True)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.title('{}'.format('Costmaps'))
    # fig.show()