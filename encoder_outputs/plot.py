import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def linear_regression(x, y):
    linear_reg = LinearRegression()
    linear_reg.fit(x, y)
    prediction = linear_reg.predict(x)
    intercept = linear_reg.intercept_
    slope = linear_reg.coef_[0]
    
    R2 = linear_reg.score(x, y)
    return prediction, intercept, slope, R2


def plot(figpath='./encoder_padding.png'):
    df = pd.read_csv('./encoder_padding.csv')
    
    prediction, intercept, slope, R2 = linear_regression(
        df[['similarity']], df['distance'])

    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    # plt.text(0.2, 0.05, 'y = {:.2f} + {:.2f} * x'.format(
    #              intercept, slope),
    #          fontsize=16, color='#ab074e')
    plt.text(0.05, 0.05, f'y={slope:.3f}x+{intercept:.3f}', 
             fontsize=16, color='#f21f62')
    plt.text(0.055, 0.03, r'$R^2$'+f'={R2:.4f}', 
             fontsize=16, color='#f21f62')

    ax.plot(
        df['similarity'],
        df['distance'],
        'ro',
        color='#4706d4',
        markersize=2.8
        )
    ax.plot(
        df['similarity'],
        prediction,
        color='#f21f62'
    )
    
    ax.set_xlabel("Tanimoto similarity", fontsize=22)
    ax.set_ylabel("Distance of latent space", fontsize=22)
    
    ax.xaxis.set_tick_params(width=2, labelsize=18)
    ax.yaxis.set_tick_params(width=2, labelsize=18)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    

plot()