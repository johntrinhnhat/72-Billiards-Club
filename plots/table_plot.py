import streamlit as st
import matplotlib.pyplot as plt
def table_plot(df_table):
    #Histogram for Check-In times
    plt.figure(figsize=(10, 4))
    plt.hist(df_table['Check_In'].apply(lambda x: x.hour), bins=24, range=(0, 24), color='skyblue', edgecolor='black')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Check-In Times')
    plt.xticks(range(0, 25))
    st.pyplot(plt)
    plt.clf()

    #Histogram for Check-Out times
    plt.figure(figsize=(10, 4))
    plt.hist(df_table['Check_Out'].apply(lambda x: x.hour), bins=24, range=(0, 24), color='salmon', edgecolor='black')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Check-Out Times')
    plt.xticks(range(0, 25))
    st.pyplot(plt)
    plt.clf()

    #Boxplot for Duration times
    plt.figure(figsize=(10, 4))
    plt.boxplot(df_table['Duration(min)'], vert=False, patch_artist=True, meanline=True, showmeans=True)
    plt.xlabel('Duration (min)')
    plt.title('Boxplot of Duration Times')
    st.pyplot(plt)
    plt.clf()

    #Scatter plot for Check-In and Check-Out Patterns
    plt.figure(figsize=(12, 6))
    plt.scatter(df_table.index, df_table['Check_In_Minutes'], alpha=0.6, label='Check-In', color='blue')
    plt.scatter(df_table.index, df_table['Check_Out_Minutes'], alpha=0.6, label='Check-Out', color='red')
    plt.xlabel('Index')
    plt.ylabel('Minutes past midnight')
    plt.title('Check-In and Check-Out Patterns')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()