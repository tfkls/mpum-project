import data_loader

if __name__ == '__main__':
    data = data_loader.load_data()
    df_train, df_test = data_loader.split_data(data, sample=[(30, 360), (21, 40)], test_bound=40)
    print(df_train)
    print(df_test)