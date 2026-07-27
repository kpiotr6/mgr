
PER_TARGET_CONFIG = {
  "return": {
    "past": [
      "ashes_feedrate",
      "auger_5_load",
      "deduster_vacuum",
      "elevator_1_load",
      "first_chamber_filling",
      "gran1_cv",
      "output_devices_power",
      "output_vacuum",
      "second_chamber_filling",
      "separator_input_temp",
      "separator_load",
      "water"
    ],
    "input": [
      "circulation_fan_speed",
      "clinker_1_feedrate",
      "fresh_feed_setpoint",
      "separator_speed",
      "stone_slag_feedrate"
    ],
  },
  "first_chamber_filling": {
    "past": [
      "ashes_feedrate",
      "auger_5_load",
      "input_sum",
      "separator_load"
    ],
    "input": [
      "aspiration_fan_speed",
      "circulation_fan_speed",
      "clinker_1_feedrate",
      "fresh_air_flap_position",
      "separator_speed",
      "stone_slag_feedrate"
    ]
  },
  "second_chamber_filling": {
    "past": [
      "first_chamber_filling",
      "gran1_vol_gt_100_um",
      "mill_output_temp",
      "return",
      "separator_input_temp",
      "water"
    ],
    "input": [
      "circulation_fan_speed",
      "fresh_feed_setpoint",
      "stone_slag_feedrate"
    ]
  },
  # "gran1_blain": {
  #   "past": [
  #     "deduster_vacuum",
  #     "gran1_cv",
  #     "gran1_davg_surf",
  #     "gran1_dv_50",
  #     "gran1_vol_gt_100_um",
  #     "gran1_vol_lt_5_um",
  #     "gran2_blain",
  #     "iron_sulfate",
  #     "mill_output_temp",
  #     "output_vacuum",
  #     "separator_power",
  #     "utility_devices_power"
  #   ],
  #   "input": [
  #     "aspiration_fan_speed",
  #     "circulation_fan_speed",
  #     "fresh_feed_setpoint",
  #     "separator_speed"
  #   ]
  # }
}


BOUNDS = {
  "return": {
    "min": 0
  },
  "first_chamber_filling": {
    "min": 0,
    "max": 100
  },
  "second_chamber_filling": {
    "min": 0,
    "max": 100
  },
  "gran1_blain": {
    "min": 0
  }
}

INPUT_COLS = [
  "circulation_fan_speed",
  "fresh_feed_setpoint",
  "aspiration_fan_speed",
  "clinker_1_feedrate",
  "fresh_air_flap_position",
  "separator_speed",
  "stone_slag_feedrate",
]

TARGET_COLS = [
  "return",
  "first_chamber_filling",
  "second_chamber_filling",
]

PAST_COLS = [
  "deduster_vacuum",
  "elevator_1_load",
  "gran1_cv",
  "output_devices_power",
  "output_vacuum",
  "ashes_feedrate",
  "auger_5_load",
  "input_sum",
  "separator_load",
  "gran1_vol_gt_100_um",
  "mill_output_temp",
  "separator_input_temp",
  "water"
]

TIME_COL = "sequence_index"
DEFAULT_FREQ = '1'

SESSION_COL = "session_index"

RAW_COLS = [
    "time",
    "sequence_index",
    "session_index",
    "cement_id",
    "clinker_1_feedrate",
    "clinker_1_feedrate_setpoint",
    "clinker_2_feedrate",
    "clinker_2_feedrate_setpoint",
    "stone_slag_feedrate",
    "stone_slag_feedrate_setpoint",
    "gypsum_feedrate",
    "gypsum_feedrate_setpoint",
    "ashes_feedrate",
    "iron_sulfate",
    "glycol",
    "water",
    "water_setpoint",
    "mill_output_temp",
    "mill_output_temp_setpoint",
    "mill_input_temp",
    "output_vacuum",
    "output_gas_flow",
    "first_chamber_filling",
    "second_chamber_filling",
    "elevator_1_load",
    "deduster_vacuum",
    "separator_load",
    "separator_speed",
    "separator_speed_setpoint",
    "separator_output_temp",
    "separator_input_temp",
    "input_sum",
    "fresh_feed_setpoint",
    "return",
    "circulation_fan_speed",
    "circulation_fan_speed_setpoint",
    "filter_fan_speed",
    "filter_fan_speed_setpoint",
    "aspiration_fan_speed",
    "aspiration_fan_speed_setpoint",
    "fresh_air_flap_position",
    "fresh_air_flap_position_setpoint",
    "auger_5_load",
    "gran1_obscuration",
    "gran1_dv_10",
    "gran1_dv_50",
    "gran1_dv_90",
    "gran1_vol_gt_100_um",
    "gran1_vol_gt_50_um",
    "gran1_vol_lt_32_um",
    "gran1_vol_in_5_50_um",
    "gran1_cv",
    "gran1_blain",
    "gran1_davg_vol",
    "gran1_davg_surf",
    "gran1_vol_lt_5_um",
    "gran2_obscuration",
    "gran2_dv_10",
    "gran2_dv_50",
    "gran2_dv_90",
    "gran2_vol_gt_100_um",
    "gran2_vol_gt_50_um",
    "gran2_vol_lt_32_um",
    "gran2_vol_in_5_50_um",
    "gran2_cv",
    "gran2_blain",
    "gran2_davg_vol",
    "gran2_davg_surf",
    "gran2_vol_lt_5_um",
    "mill_motor_power",
    "filter_power",
    "utility_devices_power",
    "output_devices_power",
    "separator_power",
    "circulation_fan_power",
]

SIMPLE_MODEL_CONFIG = {
    "target_cols": TARGET_COLS,
    "input_cols": [
        "clinker_1_feedrate",
        "separator_speed",
    ]
}


