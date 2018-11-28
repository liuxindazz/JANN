from __future__ import print_function
import visdom 
import numpy as np
import torch
#print("Hello World")

torch.nn.MarginRankingLoss
vis = visdom.Visdom(env='dann_mnist')
vis.text('hello world', win='text1')
vis.text('Hi', win='text1', append=True)


trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

# line updates
win = vis.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                        np.linspace(5, 10, 10) + 5)),
)
vis.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                        np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
vis.line(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name='2',
    update='append'
)
vis.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='delete this',
    update='append'
)
for i in xrange(10000000):  
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([i**2]), win='loss', update='append' if i> 0 else None)