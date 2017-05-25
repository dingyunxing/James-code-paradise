'''This program aims to set up a multiple regression model
   The whole project implementation by top-down design
   It includes the model construction and the prediction
   Main functions includes:
   1.Import and open csv file and read it, all variables are numbers
   2.Define one response variable and some other control variables
   3.Use ordinary least square algorithm to optimal solution
   4.Based on cross-validation
   5.Predict the response variables by the optimal solution
   
   By Yunxing Ding
'''

import numpy as np
import random
import time


def read_data(filename):
    '''the function to read file'''
    datafile = open(filename)
    data = datafile.readlines()
    datafile.close()
    return data


def open_file():
    """get file name and keep asking until filename is 
    valid and file is opened
    """

    opened = False
    while not opened:
        filename = input('Please input the name of file: ')
        try:
            file = read_data(filename)
            opened = True
        except FileNotFoundError:
            print("Can't find the file! Don't fool me!")
            opened = False
            
    return file


def process_data(data):
    '''a function to process the data'''
    list_content = []
    for line in data:
        columns = line.strip().split(',')
        list_content.append(columns)
    return list_content


def print_file_head(file, n=5):
    '''display the file head, default is first 5 line'''
    for item in file[:n]:
        print(item, end='\n')
        

def get_y_num(dataframe, y_variable):
    '''a funtion return the number of y column'''
    for num in range(len(dataframe[0])):
        if dataframe[0][num] == y_variable:
            n = num
    return n


def get_y_matrix(dataframe, n):
    '''a function to get the y list'''
    y_list = []
    
    for i in range(1,len(dataframe)):
        y_list.append(float(dataframe[i][n]))
    
    return y_list


def y_input_operate(dataframe):
    '''a function to recieve the y_variable input'''
    value = False
    
    while not value:
        y_variable = input("Please input the name of response variable: ")
        if y_variable in dataframe[0]:
            n = get_y_num(dataframe, y_variable)
            y_matrix = get_y_matrix(dataframe, n)            
            value = True
        else:
            print("please input the right name!")
            value = False
            
    return y_matrix, y_variable


def get_x_variable(dataframe, x_variable):
    '''a function to get the x comlums'''
    for num in range(len(dataframe[0])):
        if dataframe[0][num] == x_variable:
            n = num
    return n



def x_num_list_generate(dataframe):
    '''a function to recieve the x varialbe input'''
    x_num_list = []
    x_name_list = []
    lenth = len(dataframe[0])
    while lenth > 2:
        # one position for y and the other for time
        x_variable = input("please input the name of control vairalbe: ")
        if x_variable not in dataframe[0]:
            print("Please input the right name!")
        else:
            n = get_x_variable(dataframe, x_variable)
            x_num_list.append(n)
            x_name_list.append(x_variable)
            lenth -= 1            
    
    return x_num_list, x_name_list
    

def get_x_matrix(dataframe, x_num_list):
    '''a function to get the final whole x_matirx'''
    x_matrix = []
    for i in range(1,len(dataframe)):
        x_list = [1]   
        # add 1 to each row of the dataset to generate a constant column
        for j in x_num_list:
            x_list.append(float(dataframe[i][j]))
        x_matrix.append(x_list)
    
    return x_matrix    


def remove_index(dataframe, index_test):
    '''a function to remove the index of test set after split the test set.
    This will avoid the replicate of test set and validate set
    '''
    lenth = len(dataframe) - 1
    whole_list = list(range(lenth))
    for i in index_test:
        whole_list.remove(i)

    return whole_list  


        
def dataset_split(dataframe, random_test=0.2, random_validate=0.1):  
    '''a function that split data frame into training set, test set and
    validate set, return the serial number:
    
    Training set is used to train the model
    Test set is used to test perforamce
    validate set is used to validate if there are overfitting exits
    
    The default part is 7:2:1
    '''
    whole_lenth = len(dataframe) - 1
    test_index = random.sample(range(whole_lenth),\
                               round(whole_lenth*random_test))
    
    rest_index = remove_index(dataframe, test_index)
    rest_lenth = len(rest_index)
    validate_index = random.sample(range(rest_lenth),\
                                   round(rest_lenth*random_validate))
    
    list_train = []  
    list_test = []
    list_validate = []
     
    for item in range(whole_lenth):  
        if item in test_index:  
            list_test.append(item)  
        elif item in validate_index:
            list_validate.append(item)
        else:
            list_train.append(item)
          
    return list_train,list_test,list_validate
    

def proportion_judge():
    '''a function that give judge to the proportion input
       return the proportion to test and validation set respectively
    '''
    f = False
    while not f:
        p1 = float(input("Please set the test proportion(0-0.5): "))
        p2 = float(input("Please set the validate proportion(0-0.5)." +"\n" + "(Notice:validate proportion usually should be smaller than 0.2): "))
        if p1 <= 0 or p2 <= 0 or (p1 + p2) >= 1:
            print("Invalid input! What are you doing?!")
            f = False
        else:
            f = True
    return p1, p2


def set_proportion(dataframe):
    '''a funtion that asks for proportion setting up
       The return will be the training set, the test set
       and the validation set based on the user's input
    '''
    print("The default proportion of train set, test set and validate set is 0.7:0.2:0.1.")
    setup = False
    while not setup:
        
        x = input("Do you want to set the proportion yourself?(y/n) ")
        if x == 'y':
            p1, p2 = proportion_judge()
            serial_train = dataset_split(dataframe,p1,p2)[0]
            serial_test = dataset_split(dataframe,p1,p2)[1]
            serial_validate = dataset_split(dataframe,p1,p2)[2]                  
            setup = True    
                
        elif x == 'n':
            print('You are too lazy!')
            print('Ok, I will do it by default')
            serial_train = dataset_split(dataframe)[0]
            serial_test = dataset_split(dataframe)[1]
            serial_validate = dataset_split(dataframe)[2]            
            setup = True
        
        else:
            print('Invalid input! Please input it again.(y/n)')
            setup = False
    
    return serial_train, serial_test, serial_validate        


def y_split_matrix(y_matrix, serial):
    '''a function that split the y matrix according to the serial
    when serial is train, the y_split is train set
    when serial is test, the y_split is test set
    when serial is valiadation, the y_split is validation set
    '''
    y_split = []
    
    for i in serial:
        y_split.append(y_matrix[i])
        
    return np.array(y_split)
    

def x_split_matrix(x_matrix, serial):
    '''a function that split the y matrix according to the serial
    when serial is train, the y_split is train set
    when serial is test, the y_split is test set
    when serial is valiadation, the y_split is validation set
    '''    
    x_split = []
    for i in serial:
        x_split.append(x_matrix[i])
        
    return np.array(x_split)


def OLS(y,x):
    '''a functon shows the ordinary least squares algorithm to calculate the
    coefficient of each feature
    return beta, which is a matrix of coefficient
    '''
    
    x_trans = x.T
    x_square_mat = x_trans.dot(x)
    x_trans_y = x_trans.dot(y)
    beta = (np.mat(x_square_mat).I).dot(x_trans_y)
    # formular: beta = (X.T*X)-1 * X.T *Y
    
    return beta


def test_diff(y, x, y_test, x_test):
    '''a funtion to calculate the the difference on test set.
    difference = Σ(Y-beta*X) ** 2
    return a tuple with difference and coefficient
    '''
    coeff = OLS(y,x)
    diff = 0
    for i in range(len(y_test)):
        diff += (y_test[i] - (coeff.dot(x_test[i].T))) ** 2
    return (diff, coeff)    


ERROR = 9999999999
# initialise the error with a very large number

def test_times(dataframe, y_matrix, x_matrix, n=10):
    '''a function that calculate the best coefficent'''
    
    error1 = ERROR
    coef1 = []
    while n > 0:
        
        serial_train = dataset_split(dataframe)[0]
        serial_test = dataset_split(dataframe)[1]
        
        x = x_split_matrix(x_matrix, serial_train)
        y = y_split_matrix(y_matrix, serial_train)        
        y_test = y_split_matrix(y_matrix, serial_test)
        x_test = x_split_matrix(x_matrix, serial_test)
        
        error2, coef2 = test_diff(y, x, y_test, x_test)
        
        if error2 < error1:
            error1 = error2
            coef1 = coef2

        else:
            n -= 1
    
    return coef1, error1


def validate_best(dataframe, y_matrix, x_matrix):
    '''a function that validate the difference on validate set'''
    serial_validate = dataset_split(dataframe)[2]
    
    x = x_split_matrix(x_matrix, serial_validate)
    y = y_split_matrix(y_matrix, serial_validate)
    
    diff = 0
    coeff = test_times(dataframe, y_matrix, x_matrix)[0]
    for i in range(len(y)):
        diff += (y[i] - (coeff.dot(x[i].T))) ** 2
    
    return diff


def count_time(m):
    '''a countdown function'''
    count = 0
    while (count < m):
        
        count += 1
        n = m - count
        time.sleep(1)
        print(n + 1, "times left")        


def satisfy_or_not(dataframe, y_matrix, x_matrix, coef1, sst1, error1, n=10):
    '''a funtion that judge if the user is satisfy with the total sum square'''
    satisfy = False
    coef = coef1
    sst = sst1   
    error = error1
    while not satisfy:

        judge = input("Do you satisfiy to the validation result? (y/n) ")
        if judge == "n":
            print('*' * 80)
            coef1, error1 = test_times(dataframe, y_matrix, x_matrix, n)
            sst = validate_best(dataframe, y_matrix, x_matrix)
            count_time(n)
            print("The updated SST on test set is:", error1)
            print("The updated SST on validation set is:", sst)
            satisfy = False
        elif judge == "y":
            print('*' * 80)    
            satisfy = True
        else:
            print("The input is invalid, please input again. (y/n)")
            satisfy = False
    
    print("The final best coefficient matrix after {} times validation is:"\
          .format(n) + '\n', coef1)
    print("The SST on validation set is", sst)            
    
    return coef1


def get_predict_x(dataframe, x_name_list):
    '''a function that get the predicted value of responce varaible'''
    
    list_predict_x = [1]
    list_name = x_name_list
    x = 0
    for i in list_name:
        
        x = float(input("Please input the value of {}: ".format(i)))
        list_predict_x.append(x)
        
    return list_predict_x
        

def quit_or_predict(dataframe, x_name_list, y_variable, c):
    '''a function asks whether continue predict or quit'''
    
    quit = False
    while not quit:
        a = input("Do you want to continue prediction or quit the programme?(continue/quit) ")
        if a == 'continue':
            print('\n' * 2)
            list1 = get_predict_x(dataframe, x_name_list)
            predict_y = c.dot(np.array(list1).reshape([len(list1),1]))
            print("the final predict {} value is:".\
                  format(y_variable), predict_y)
            quit = False
        elif a == 'quit':
            print('*' * 80)
            print("Thank you for using this amazing program!!!")
            print("See you next time!")
            quit = True
        else:
            print("The input is invalid, please input again!(continue/quit)")
            quit = False




# **********************************************************************
#
# Now at last the main function and the call to it
#
# **********************************************************************

def main():
    '''the main function'''
    print("Welcome to our magic world!^_^" + "\n")
    file = open_file()
    dataframe = process_data(file)
    
    times = int(input("How many rows do you want the dataframe to show?(at least 2) "))
    print_file_head(dataframe, times)
    
    y_matrix, y_variable = y_input_operate(dataframe)
    x_num_list, x_name_list = x_num_list_generate(dataframe)
    x_matrix = get_x_matrix(dataframe, x_num_list)
    
    serial_train, serial_test, serial_validate = set_proportion(dataframe)
    
    x = x_split_matrix(x_matrix, serial_train)
    y = y_split_matrix(y_matrix, serial_train)     
    
    n= int(input("How many times do you want to do the cross validation? (1-50)"))
    count_time(n)
    coef1, error1 = test_times(dataframe, y_matrix, x_matrix, n)
    sst1 = validate_best(dataframe, y_matrix, x_matrix)
    print("The final best coefficient matrix after {} times validation is:".format(n) + '\n', coef1)
    
    print("*" * 80)
    print("The SST on test set is:", error1)
    print("The SST on validation set is:", sst1)
    
    c = satisfy_or_not(dataframe, y_matrix, x_matrix, coef1, sst1, error1, n)        
    # c is the final coefficient matrix can be used in prediction
    
    print('*' * 80)
    print("Now we got the model, we can begin to predict!!!")
    print('\n' * 2)
    
    list_input = get_predict_x(dataframe, x_name_list)
    
    predict_y = c.dot(np.array(list_input).reshape([len(list_input),1]))
    print('*' * 80)
    print("the final predict {} value is:".format(y_variable), predict_y)
    
    quit_or_predict(dataframe, x_name_list, y_variable, c)


main()


