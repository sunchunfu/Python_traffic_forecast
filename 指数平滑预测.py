import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  

def get_once_predict(history_value, smoothed_value, alpha, decimal_point=0):
    '''
    获取平滑后预测值
    :param history_value: 历史值
    :param smoothed_value: 平滑值
    :param alpha: 加权系数
    :param decimal_point: 小数点位数
    :return:
    '''
    if decimal_point == 0:
        return round(alpha * history_value + (1 - alpha) * smoothed_value)
    return round(alpha * history_value + (1 - alpha) * smoothed_value, decimal_point)


def once_smoothing(history_data, alpha, s0=None, decimal_point=0):
    '''
    预测一次指数平滑数据
    :param history_data: 历史数据
    :param alpha: 加权系数
    :param s0: 初始值
    :param decimal_point: 小数点位数
    :return: 一次平滑结果
    '''
    history_data_len = len(history_data)
    if s0 == None:
        # 如果不给定始值，就取前三个值均值
        if 3 < history_data_len:
            s0 = sum(history_data[:3]) / 3
        elif 1 < history_data_len:
            s0 = history_data[0]
        else:
            return history_data
    s = [s0]
    smoothed_value = s0
    for data in history_data:
        smoothed_value = get_once_predict(data, smoothed_value, alpha, decimal_point=decimal_point)
        s.append(smoothed_value)
    return s


def second_smoothing(once_data, alpha, decimal_point=0):
    '''
    预测二次指数平滑数据
    :param once_data: 一次指数平滑后结果
    :param alpha: 加权系数
    :return: 二次平滑结果
    '''
    second_data = once_smoothing(once_data[1:], alpha, once_data[0], decimal_point)
    return second_data


def get_second_predict_n(n, s1, s2, alpha, decimal_point=0):
    '''
    获取二次指数平滑第n个预测值
    :param n: 未来第n个值
    :param s1: 一次指数平滑最后的值
    :param s2: 二次指数平滑最后的值
    :param alpha: 加权系数
    :param decimal_point: 小数点位数
    :return: 第n个预测值
    '''

    a = 2 * s1 - s2
    b = alpha / (1 - alpha) * (s1 - s2)
    # y = round(a + b * n, 2)
    if decimal_point == 0:
        y = round(a + b * n)
    else:
        y = round(a + b * n, decimal_point)
    return y


def third_smoothing(second_data, alpha, decimal_point=0):
    '''
    预测三次指数平滑数据
    :param second_data: 二次指数平滑后结果
    :param alpha: 加权系数
    :param decimal_point: 小数点位数
    :return:
    '''
    second_data = second_smoothing(second_data, alpha, decimal_point)
    return second_data


def get_third_predict_n(n, s1, s2, s3, alpha, decimal_point=0):
    '''
    获取三次平滑第n个预测值
    :param n: 未来第n个
    :param s1: 一次指数平滑最后的值
    :param s2: 二次指数平滑最后的值
    :param s2: 三次指数平滑最后的值
    :param alpha: 加权系数
    :param decimal_point: 小数点位数
    :return: 第n个预测值
    '''
    a = round(3 * s1 - 3 * s2 + s3, 2)
    b = round(
        (alpha / (2 * pow(1 - alpha, 2))) * ((6 - 5 * alpha) * s1 - 2 * (5 - 4 * alpha) * s2 + (4 - 3 * alpha) * s3), 2)
    c = round(pow(alpha, 2) / (2 * pow(1 - alpha, 2)) * (s1 - 2 * s2 + s3), 2)
    if decimal_point == 0:
        y = round(a + b * n + c * pow(n, 2))
    else:
        y = round(a + b * n + c * pow(n, 2), decimal_point)
    print('非线性预测模型的系数a,b,c', a, b, c)
    return y

def pre_next(history_data,alpha = 0.2):
    s0 = sum(history_data[:3]) / 3
    once_data = once_smoothing(history_data, alpha, s0, 4)
    second_data = second_smoothing(once_data, alpha, 4)
    third_data = third_smoothing(second_data, alpha,4)
    pred_data = get_third_predict_n(1, once_data[-1], second_data[-1], third_data[-1], alpha, 4)
    return pred_data,once_data[1:],second_data[1:],third_data[1:]


if __name__ == '__main__':
    df = pd.read_excel('交通数据.xlsx', sheet_name='Sheet6')
    history_data = df['日交通量'].tolist()
    pred_data,once_data,second_data,third_data=pre_next(history_data, alpha=0.5)
    pred_value = []
    for i in range(10):
        next_value = get_third_predict_n(i+1, once_data[-1], second_data[-1], third_data[-1], alpha=0.5, decimal_point=4)
        print(next_value)
        pred_value.append(next_value)

    plt.xlim(0, 360)
    plt.ylim(0, 25000)
    plt.plot(history_data, label='实际值')
    plt.plot(once_data, label='一次指数平滑')
    plt.plot(second_data, label='二次指数平滑')
    plt.plot(third_data, label='三次指数平滑')
    plt.plot([i+len(history_data) for i in range(10)], pred_value, label='预测值')
    plt.legend()
    plt.savefig('指数平滑日交通量.png', dpi=300)
    plt.show()

