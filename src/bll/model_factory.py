from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_stacked_lstm_model(
    input_shape: tuple[int, int],
    hidden_units: int,
    num_layers: int,
    dropout_rate: float,
    learning_rate: float,
) -> Sequential:
    model = Sequential()

    for layer_index in range(num_layers):
        return_sequences = layer_index < num_layers - 1
        if layer_index == 0:
            model.add(
                LSTM(
                    hidden_units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                )
            )
        else:
            model.add(LSTM(hidden_units, return_sequences=return_sequences))

        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model