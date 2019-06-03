def get_bit_dataframe(raw_data_filepath):
    """
    Insert raw txt file with data ( good for Kvaser leaf light v2. Little pre
    processing of file is necesary, will fix later)
    """
    data =pd.read_fwf(raw_data_filepath)
    #drops info which is not necessary for neural nets
    data2 = data.drop(['Dir', 'Flg', 'Chn', 'DLC', 'Time'], axis = 1)
    #renming columns
    data3 = data2.rename(columns={'Identifier' :'PID', 'D0':'B1','1' :'B2','2':'B3','3' :'B4','4':'B5','5': 'B6','6':'B7', 'D7':'B8'})
    #it looks like if byte is not used at all it gets an empty space which
    #turns into NaN during conversion from txt to dataframe.
    #For now turn NaN to 0
    data4 = data3.fillna(0)
    #drops last row which indicates end of data collection
    data5 = data4[:-1]
    data6 = data5.astype('int64')
    #convert to binary and pad with zeros in front for uniform input
    data7 = data6.iloc[:,1:].applymap(lambda x: bin(x)[2:].zfill(8))
    data7['PID'] = data6['PID']
    data7['Data'] = data7['B1'] + data7['B2'] + data7['B3'] + data7['B4'] + data7['B5'] + data7['B6'] + data7['B7'] + data7['B8']
    for i in range(64):
        data7['b'+str(i+1)] = data7['Data'].str[i]
    # data8 = data7.iloc[:,8:].drop(['Data'], axis =1)
    return data7

def pid_for_training(data_frame, pid = 494):
    """
    Input is a dataframe obtained from function get_bit_dataframe and which
    pid to consider. Output is n x 64 dataframe (n signals of said pid, 64 bit
    message for each signal)
    """
    df_pid = data_frame.loc[data_frame['PID'] == pid].drop(['PID', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'Data'], axis =1).reset_index(drop=True)
    return df_pid


def matrix_for_rnn_training(data_frame, timestep = 10, shift =1 ):
    """
    input is n x m pandas dataframe (m=number of features (64* #of_pid),
    n = number of samples)
    """
    M = np.array(data_frame)
    x_y_tups = [(M[i:i+ timestep], M[i+timestep]) for i in np.arange(0,M.shape[0]-timestep, shift)]
    X_list, Y_list = list(zip(*x_y_tups))
    X=np.dstack(X_list).swapaxes(0,2).swapaxes(1,2)
    Y=np.array(Y_list)
    return X, Y
