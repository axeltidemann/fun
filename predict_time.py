import calendar
import datetime
import argparse

from tensorflow import keras
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--time_encoding',
                    choices=['embedding', 'continuous'],
                    default='embedding')
parser.add_argument('--print_dates',
                    action='store_true')
parser.add_argument('--embed_size',
                    type=int,
                    default=64)
parser.add_argument('--hidden_size',
                    type=int,
                    default=32)
parser.add_argument('--layers',
                    type=int,
                    default=3)
parser.add_argument('--dropout_rate',
                    type=float,
                    default=0)
parser.add_argument('--epochs',
                    type=int,
                    default=25)
parser.add_argument('--batch_size',
                    type=int,
                    default=8)
parser.add_argument('--noise',
                    type=int,
                    default=0,
                    help=('The number of noisy channels (uniform). '
                          'Noise helps, probably acts as a regularizer.'))
parser.add_argument('--days_ahead',
                    type=int,
                    default=1,
                    help='The number of days ahead in the future to predict the date.')
args = parser.parse_args()

now = datetime.datetime.now()

def month_days(days, month):
    m_d = np.empty((days, 2))
    m_d[:,0] = month
    m_d[:,1] = np.arange(days)
    return m_d

X = [ month_days(calendar.monthrange(now.year, i+1)[1], i) for i in range(12) ]
X = np.concatenate(X, axis=0)

y = np.roll(X, -args.days_ahead, axis=0)

X_in = {'month_in': X[:,0], 'day_in': X[:,1]}
y_out = {'month_out': y[:,0], 'day_out': y[:,1]}

if args.noise:
    X_in['noise'] = np.random.uniform(low=-1, high=1, size=(X.shape[0], args.noise))

if args.print_dates:
    for today,next_date in zip(X,y):
        print('today',today,'next_date',next_date)

month_input = keras.Input(shape=(1,), name='month_in')
day_input = keras.Input(shape=(1,), name='day_in')
noise_input = keras.Input(shape=(args.noise,), name='noise')

inputs = [month_input, day_input]
if args.noise:
    inputs.append(noise_input)

if args.time_encoding == 'embedding':
    month_embedding = keras.layers.Embedding(input_dim=12, output_dim=args.embed_size,
                                             input_length=1)(month_input)
    day_embedding = keras.layers.Embedding(input_dim=31, output_dim=args.embed_size,
                                           input_length=1)(day_input)
    month_flatten = keras.layers.Flatten()(month_embedding)
    day_flatten = keras.layers.Flatten()(day_embedding)
else:
    month_flatten = month_input
    day_flatten = day_input

    X[:,0] = X[:,0]/11
    X[:,1] = X[:,1]/30

embeddings = [ month_flatten, day_flatten ]
if args.noise:
    embeddings.append(noise_input)
                                      
x = keras.layers.Concatenate()(embeddings)

for _ in range(args.layers):
    x = keras.layers.Dense(args.hidden_size, activation='elu')(x)
    x = keras.layers.Dropout(args.dropout_rate)(x)

month_output = keras.layers.Dense(12, activation='softmax', name='month_out')(x)    
day_output = keras.layers.Dense(31, activation='softmax', name='day_out')(x)

model = keras.Model(inputs=inputs, outputs=[month_output, day_output])

model.compile(metrics='accuracy', loss='sparse_categorical_crossentropy', optimizer='adam')

print(model.summary())

model.fit(X_in, y_out, validation_data=(X_in, y_out),
          epochs=args.epochs, batch_size=args.batch_size)

month_predict, day_predict = model.predict(X_in)

month_predict = np.argmax(month_predict, axis=1)
day_predict = np.argmax(day_predict, axis=1)

prediction = np.empty((len(month_predict),2))
prediction[:,0] = month_predict
prediction[:,1] = day_predict

for today, next_date, _prediction in zip(X, y, prediction):
    if not np.all(next_date == _prediction):
        print('today',today,'next_date',next_date,'prediction',_prediction)
