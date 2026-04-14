import numpy as np
import pandas as pd
import itertools
from darts import TimeSeries

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_COLS, TARGET_COLS

class DiscreteMPCController:
    """
    A Model Predictive Controller (MPC) that evaluates a discrete set
    of possible next-step control inputs using a trained Darts model.
    """
    def __init__(self, model, horizon, allowed_input_values):
        """
        :param model: The trained Darts model capable of forecasting targets.
        :param horizon: The prediction/control horizon (number of steps to look ahead).
        :param allowed_input_values: A dictionary mapping each column in INPUT_COLS
                                     to a list of allowed discretized values (e.g., up to 3).
        """
        self.model = model
        self.horizon = horizon
        self.allowed_input_values = allowed_input_values

        # Make sure all INPUT_COLS have defined allowed values
        for col in INPUT_COLS:
            if col not in self.allowed_input_values:
                raise ValueError(f"Missing allowed values for input column: {col}")

        # Generate all combinations (e.g., 3^4 = 81 steering possibilities)
        keys = INPUT_COLS
        values = [self.allowed_input_values[k] for k in keys]
        self.action_space = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    def get_best_next_inputs(self, desired_targets, past_targets, past_covariates=None):
        """
        Tests all combinations of next-step inputs. Evaluates the outputs against
        desired_targets using Mean Squared Error (MSE).
        Returns the best next step input combination.

        :param desired_targets: The target values we want to achieve over the horizon.
                                This should be an array broadcastable to the model's output shape (horizon, num_targets).
        :param past_targets: A Darts TimeSeries containing past target values.
        :param past_covariates: A Darts TimeSeries containing past covariates (if the model requires them).
        :return: A tuple (best_action, best_mse) where best_action is a dictionary of the best input values.
        """
        best_mse = float('inf')
        best_action = None

        desired_targets_arr = np.array(desired_targets)

        # Calculate appropriate future time index to avoid Darts index alignment issues
        if isinstance(past_targets.time_index, pd.DatetimeIndex):
            freq = past_targets.freq
            start_idx = past_targets.time_index[-1] + freq
            time_index = pd.date_range(start=start_idx, periods=self.horizon, freq=freq)
        else:
            # Assuming RangeIndex or numeric index
            step = past_targets.time_index[-1] - past_targets.time_index[-2] if len(past_targets) > 1 else 1
            start_idx = past_targets.time_index[-1] + step
            time_index = pd.RangeIndex(start=start_idx, stop=start_idx + self.horizon * step, step=step)

        for action in self.action_space:
            # For each combination, generate discrete values that stay constant over the horizon
            action_row = [action[col] for col in INPUT_COLS]
            future_inputs_data = [action_row for _ in range(self.horizon)]

            df_future = pd.DataFrame(future_inputs_data, columns=INPUT_COLS, index=time_index)
            future_covariates_ts = TimeSeries.from_dataframe(df_future)

            # Prepare predict kwargs
            predict_kwargs = {
                'n': self.horizon,
                'series': past_targets,
                'future_covariates': future_covariates_ts
            }
            if past_covariates is not None:
                predict_kwargs['past_covariates'] = past_covariates

            try:
                prediction = self.model.predict(**predict_kwargs)
                predicted_vals = prediction.values()

                # Calculate MSE (Mean Squared Error) between prediction and desired targets
                mse = np.mean((predicted_vals - desired_targets_arr) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_action = action
            except Exception as e:
                # E.g., model predict failure
                print(f"Prediction failed for action {action}: {e}")
                continue

        return best_action, best_mse

if __name__ == "__main__":
    from config import TARGET_COLS

    class DummyModel:
        def predict(self, n, series=None, past_covariates=None, future_covariates=None):
            # Output shape: (n, len(TARGET_COLS))
            index = future_covariates.time_index if future_covariates is not None else range(n)
            data = np.random.rand(n, len(TARGET_COLS))
            return TimeSeries.from_times_and_values(index, data, columns=TARGET_COLS)

    allowed_input_values = {
        "separator_speed": [500, 600, 700],
        "fresh_feed_setpoint": [50, 75, 100],
        "circulation_fan_speed": [600, 700, 800],
        "aspiration_fan_speed": [800, 900, 1000]
    }

    mock_model = DummyModel()
    horizon = 5
    controller = DiscreteMPCController(mock_model, horizon, allowed_input_values)

    past_index = pd.RangeIndex(start=0, stop=10, step=1)
    past_data = np.random.rand(10, len(TARGET_COLS))
    past_targets = TimeSeries.from_times_and_values(past_index, past_data, columns=TARGET_COLS)

    # Desired targets for the next 'horizon' steps - shape must match (horizon, num_targets)
    desired_targets = np.full((horizon, len(TARGET_COLS)), 0.5)

    best_action, best_mse = controller.get_best_next_inputs(desired_targets, past_targets)

    print(f"Best Action: {best_action}")
    print(f"Best MSE: {best_mse}")
