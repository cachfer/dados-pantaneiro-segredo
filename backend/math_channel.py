# author: carolina ferrari

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy.integrate import cumulative_trapezoid as cumtrapz
from constants import (g, rho, cl, cd, wb, ft, rt, fa, wf, wr, w, steering_ratio, rl, gear_reduction)
from data_configuration import COLUMN_MAPPING, REQUIRED_COLUMNS, THRESHOLDS, convert_to_si, convert_from_si

def code_variables(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=',', engine='python', decimal='.')
    df = df.rename(columns=COLUMN_MAPPING)
    df_cleaned = pd.DataFrame()
    df_cleaned['time'] = df['time'] - df['time'].iloc[0] # t_0 = 0

    # add required columns, using NaN for missing ones
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df_cleaned[col] = df[col]
        else:
            print(f"Warning: '{col}' column is missing. Initializing with NaN.")
            df_cleaned[col] = np.nan
    return df_cleaned

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    process the cleaned dataframe to compute derived metrics and indicators
    arguments:
        df: input dataframe with required columns  
    returns:
        pd.DataFrame: processed dataframe with computed metrics
    """
    try:
        # convert columns to numeric, handling any non-numeric values
        for col in REQUIRED_COLUMNS: 
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # check for excessive NaN values
                nan_percentage = df[col].isna().sum() / len(df) * 100
                if nan_percentage > 50:
                    print(f"Warning: Column '{col}' has {nan_percentage:.1f}% NaN values")
        
        # create a working copy of the data
        data = {}
        # copy all required columns from dataframe
        for col in REQUIRED_COLUMNS:
            data[col] = df[col].values
        
        # CONVERT INPUT DATA TO SI UNITS FOR CALCULATIONS
        si_data = {}
        for col in REQUIRED_COLUMNS:
            if col in data:
                si_data[col] = convert_to_si(data, col, data[col])
            else:
                si_data[col] = np.full(len(data['time']), np.nan)
        
        # calculate time-based arrays length
        data_length = len(data['time'])
        
        # validate data length consistency
        if data_length == 0:
            raise ValueError("No valid data points found")
        print(f"Processing {data_length} data points")
        
        # initialize arrays with different default values
        nan_arrays = {
            'brake_bias': np.nan,
            'steer_speed_on': np.nan,
            'steer_speed_off': np.nan,
            'brake_speed_on': np.nan,
            'brake_speed_off': np.nan,
            'throttle_speed_on': np.nan,
            'throttle_speed_off': np.nan,
            'low_oil_pressure': np.nan
        }
        zero_arrays = {
            'coasting': 0,
            'coasting_off_brakes': 0,
            'coasting_off_throttle': 0,
            'crossing': 0,
            'full_throttle': 0,
            'trail_braking': 0,
            'GF_aero': 0,
            'GF_braking': 0,
            'GF_cornering': 0,
            'GF_overall': 0,
            'GF_traction': 0,
            'understeering': 0,
            'oversteering': 0,
            'number_gear_shifts': 0,
            'steer_throttle': 0,
            'torque': 0
        }
        one_arrays = {
            'driver_inactive': 1
        }
        # initialize all arrays at once
        data.update({k: np.full(data_length, v) for k, v in nan_arrays.items()})
        data.update({k: np.zeros(data_length) for k, v in zero_arrays.items()})
        data.update({k: np.ones(data_length) for k, v in one_arrays.items()})

        # MATH #########################################################

        # distance, speed, aero forces (using SI data)
        traction_speed = si_data['traction_speed']

        data['distance'] = cumtrapz(traction_speed, si_data['time'], initial=0) 
        data['G_combined'] = np.sqrt(si_data['G_lateral']**2 + si_data['G_longitudinal']**2)
        "data['downforce'] = 0.5 * rho * fa * cl * traction_speed**2"
        "data['drag'] = 0.5 * rho * fa * cd * traction_speed**2"
        brake_front_si = si_data['brake_pressure_front']
        brake_rear_si = si_data['brake_pressure_rear']
        brake_total_si = brake_front_si + brake_rear_si
        data['brake_pressure_total'] = convert_from_si(data, 'brake_pressure_total', brake_total_si)
        
        steering_angle_si = steering_ratio * si_data['wps']
        data['steering_angle'] = convert_from_si(data, 'steering_angle', steering_angle_si)
        
        power_si = data['torque'] * traction_speed / rl  # Power in W
        data['power'] = convert_from_si(data, 'power', power_si)
        
        # Initialize gear_shifts column and calculate gear shifts
        data['gear_shifts'] = np.zeros(data_length)
        
        if data_length > 1:
            data['gear_shifts'][1:] = (data['gear'][1:] != data['gear'][:-1]).astype(int)
            data['total_gear_shifts'] = np.sum(data['gear_shifts'])
        else:
            data['total_gear_shifts'] = 0

        low_oil_pressure_limit_si = np.where(si_data['rpm'] < THRESHOLDS['rpm_threshold'], 2.75/8000 * (si_data['rpm'] - 1500) + 0.75, 3.5)
        data['low_oil_pressure_limit'] = convert_from_si(data, 'low_oil_pressure_limit', low_oil_pressure_limit_si)

        data['low_oil_pressure'] = (si_data['oil_pressure'] < low_oil_pressure_limit_si).astype(int)
        data['low_fuel_pressure'] = (si_data['fuel_pressure'] <= THRESHOLDS['low_fuel_pressure']).astype(int)
        data['low_battery_voltage'] = (data['battery_voltage'] <= THRESHOLDS['low_battery']).astype(int)
        data['engine_overheating'] = (si_data['engine_temperature'] >= THRESHOLDS['high_engine_temp']).astype(int)
        data['oil_overheating'] = (si_data['oil_temperature'] >= THRESHOLDS['high_oil_temp']).astype(int)
        data['fuel_consumption'] = cumtrapz(data['fuel_flow']/3600, data['time'], initial=0) # fuel_flow is in lb/h, convert to lb/s        
        data['lambda_error'] = data['general_probe'] - data['mesh_target']

        with np.errstate(divide='ignore', invalid='ignore'):
            valid_lateral_g = ~np.isnan(si_data['G_lateral']) & (np.abs(si_data['G_lateral']) > 0.1)
        
            R_si = np.full(data_length, np.nan)
            R_si[valid_lateral_g] = traction_speed[valid_lateral_g]**2 / (np.abs(si_data['G_lateral'][valid_lateral_g]) * g)
            data['R'] = R_si  # Already in meters (SI)
            
            data['curvature'] = np.where(R_si != 0, 1 / R_si, np.nan)
            
            delta_acker_si = np.where(R_si != 0, wb / R_si, np.nan)
            data['delta_acker'] = convert_from_si(data, 'delta_acker', delta_acker_si)
            
            understeer_gradient_si = np.where(si_data['G_lateral'] != 0, (si_data['wps'] - delta_acker_si) / si_data['G_lateral'], np.nan)
            data['understeer_gradient'] = convert_from_si(data, 'understeer_gradient', understeer_gradient_si)
            
            data['yaw_rate'] = np.where(traction_speed != 0, si_data['G_lateral'] / traction_speed, np.nan)

        data['understeering'] = (data['steering_angle'] < data['delta_acker']).astype(int)
        data['oversteering'] = (data['steering_angle'] > data['delta_acker']).astype(int)
        data['neutral_steering'] = (data['steering_angle'] == data['delta_acker']).astype(int)

        if np.all(np.isnan(data['brake_pressure_total'])):
            print("Warning: All brake pressure values are NaN. Using zeros instead.")
            data['brake_pressure_total'] = np.zeros(data_length)
            
        mask = (si_data['brake_pressure_front'] > THRESHOLDS['brake_pressure_min']) & (si_data['brake_pressure_rear'] > THRESHOLDS['brake_pressure_min'])
        data['brake_bias'] = np.where(mask, data['brake_pressure_front'] / data['brake_pressure_total'] * 100, np.nan)

        aero_mask = (si_data['G_lateral'] >= THRESHOLDS['high_lateral_g']) & (si_data['traction_speed'] >= THRESHOLDS['min_speed_aero'])
        data['GF_aero'] = np.where(aero_mask, data['G_combined'], np.nan)

        brake_mask = si_data['G_longitudinal'] <= -THRESHOLDS['high_lateral_g']
        data['GF_braking'] = np.where(brake_mask, data['G_combined'], np.nan) # braking: significant negative longitudinal G

        corner_mask = si_data['G_lateral'] >= THRESHOLDS['high_lateral_g']
        data['GF_cornering'] = np.where(corner_mask, data['G_combined'], np.nan) # cornering: high lateral G

        traction_mask = (si_data['G_longitudinal'] >= THRESHOLDS['min_traction_g']) & (si_data['G_lateral'] >= THRESHOLDS['high_lateral_g'])
        data['GF_traction'] = np.where(traction_mask, data['G_combined'], np.nan) # traction: positive longitudinal G and high lateral G

        overall_mask = data['G_combined'] >= THRESHOLDS['high_g_combined']
        data['GF_overall'] = np.where(overall_mask, data['G_combined'], np.nan) # overall: high combined G
        
        "pos_accel_mask = data['G_longitudinal'] > 0"
        "data['torque'][pos_accel_mask] = (w * data['G_longitudinal'][pos_accel_mask] + data['drag'][pos_accel_mask]) * rl / 1000"
        
        steer_diff_si = np.diff(np.abs(si_data['wps'])) / np.diff(si_data['time'])
        data['steer_differential'] = convert_from_si(data, 'steer_differential', np.pad(steer_diff_si, (0,1), mode='constant', constant_values=np.nan))
              
        brake_diff_si = np.diff(si_data['brake_pressure_front']) / np.diff(si_data['time'])
        data['brake_differential'] = convert_from_si(data, 'brake_differential', np.pad(brake_diff_si, (0,1), mode='constant', constant_values=np.nan))
             
        throttle_diff_si = np.diff(si_data['tps']) / np.diff(si_data['time'])
        data['throttle_differential'] = convert_from_si(data, 'throttle_differential', np.pad(throttle_diff_si, (0,1), mode='constant', constant_values=np.nan))
        
        for i in range(1, data_length):
            if data['brake_pressure_total'][i] > 50 and abs(data['wps'][i]) > 5:
                data['trail_braking'][i] = 1
            if data['tps'][i] > 5 and abs(data['wps'][i]) > 10:
                data['steer_throttle'][i] = 1
            if data['tps'][i] >= 95:
                data['full_throttle'][i] = 1
            if data['brake_pressure_total'][i] < 50 and data['tps'][i] < 5:
                data['coasting'][i] = 1
                if data['tps'][i-1] >= 5 or data['coasting_off_throttle'][i-1] == 1:
                    data['coasting_off_throttle'][i] = 1
                elif data['brake_pressure_total'][i-1] >= 50 or data['coasting_off_brakes'][i-1] == 1:
                    data['coasting_off_brakes'][i] = 1
            if data['brake_pressure_total'][i] > 50 and data['tps'][i] > 5:
                data['crossing'][i] = 1
            if data['brake_pressure_total'][i] > 50 or data['tps'][i] > 5 or abs(data['wps'][i]) > 10:
                data['driver_inactive'][i] = 0

        # brake speed with high brake pressure (using SI data)
        high_brake_mask = si_data['brake_pressure_front'] > THRESHOLDS['high_brake_pressure']
        pos_brake_diff = brake_diff_si > 0
        neg_brake_diff = brake_diff_si < 0
        
        data['brake_speed_on'] = np.zeros(data_length)
        data['brake_speed_on'][:-1] = np.where(pos_brake_diff & high_brake_mask[:-1], convert_from_si(data, 'brake_speed_on', brake_diff_si), 0)
        data['brake_speed_off'] = np.zeros(data_length)
        data['brake_speed_off'][:-1] = np.where(neg_brake_diff & high_brake_mask[:-1], convert_from_si(data, 'brake_speed_off', np.abs(brake_diff_si)), 0)
        
        # throttle speed with significant throttle (using SI data)
        high_throttle_mask = si_data['tps'] > THRESHOLDS['low_tps']
        pos_throttle_diff = throttle_diff_si > 0
        neg_throttle_diff = throttle_diff_si < 0
        
        data['throttle_speed_on'] = np.zeros(data_length)
        data['throttle_speed_on'][:-1] = np.where(pos_throttle_diff & high_throttle_mask[:-1], convert_from_si(data, 'throttle_speed_on', throttle_diff_si), 0)
        data['throttle_speed_off'] = np.zeros(data_length)
        data['throttle_speed_off'][:-1] = np.where(neg_throttle_diff & high_throttle_mask[:-1], convert_from_si(data, 'throttle_speed_off', np.abs(throttle_diff_si)), 0)

        def front_left_damper_curve(x):
            return 1.8003 * x**3 - 4.1935 * x**2 + 9.2702 * x + 170.56
        def front_right_damper_curve(x):
            return 4.5217 * x**3 - 19.413 * x**2 + 34.887 * x + 163.02
        def rear_left_damper_curve(x):
            return 1.8003 * x**3 - 4.1935 * x**2 + 9.2702 * x + 170.56
        def rear_right_damper_curve(x):
            return 1.8003 * x**3 - 4.1935 * x**2 + 9.2702 * x + 170.56

        # Convert damper positions to SI for calculations
        FL_raw_travel_si = front_left_damper_curve(si_data['FL_Damper'])
        FR_raw_travel_si = front_right_damper_curve(si_data['FR_Damper'])
        RL_raw_travel_si = rear_left_damper_curve(si_data['RL_Damper'])
        RR_raw_travel_si = rear_right_damper_curve(si_data['RR_Damper'])

        # Calculate travel in SI units
        FL_travel_si = FL_raw_travel_si - FL_raw_travel_si[0]
        FR_travel_si = FR_raw_travel_si - FR_raw_travel_si[0]
        RL_travel_si = RL_raw_travel_si - RL_raw_travel_si[0]
        RR_travel_si = RR_raw_travel_si - RR_raw_travel_si[0]
        
        # Convert back to original units for output
        data['FL_Damper_travel'] = convert_from_si(data, 'FL_Damper_travel', FL_travel_si)
        data['FR_Damper_travel'] = convert_from_si(data, 'FR_Damper_travel', FR_travel_si)
        data['RL_Damper_travel'] = convert_from_si(data, 'RL_Damper_travel', RL_travel_si)
        data['RR_Damper_travel'] = convert_from_si(data, 'RR_Damper_travel', RR_travel_si)

        # Calculate damper speeds using SI data
        fl_travel_diff_si = np.diff(FL_travel_si) / np.diff(si_data['time'])
        fr_travel_diff_si = np.diff(FR_travel_si) / np.diff(si_data['time'])
        rl_travel_diff_si = np.diff(RL_travel_si) / np.diff(si_data['time'])
        rr_travel_diff_si = np.diff(RR_travel_si) / np.diff(si_data['time'])

        data['FL_Damper_speed'] = convert_from_si(data, 'FL_Damper_speed', np.pad(fl_travel_diff_si, (0,1), mode='constant', constant_values=np.nan))
        data['FR_Damper_speed'] = convert_from_si(data, 'FR_Damper_speed', np.pad(fr_travel_diff_si, (0,1), mode='constant', constant_values=np.nan))
        data['RL_Damper_speed'] = convert_from_si(data, 'RL_Damper_speed', np.pad(rl_travel_diff_si, (0,1), mode='constant', constant_values=np.nan))
        data['RR_Damper_speed'] = convert_from_si(data, 'RR_Damper_speed', np.pad(rr_travel_diff_si, (0,1), mode='constant', constant_values=np.nan))

        # vehicle metrics 
        data['front_roll'] = data['FR_Damper_travel'] - data['FL_Damper_travel']
        data['rear_roll'] = data['RR_Damper_travel'] - data['RL_Damper_travel']
        data['left_pitch'] = data['RL_Damper_travel'] - data['FL_Damper_travel']
        data['right_pitch'] = data['RR_Damper_travel'] - data['FR_Damper_travel']
        data['front_yaw'] = (data['FL_Damper_travel'] + data['FR_Damper_travel']) / 2
        data['rear_yaw'] = (data['RL_Damper_travel'] + data['RR_Damper_travel']) / 2
        data['total_yaw'] = (data['front_yaw'] + data['rear_yaw']) / 2        

        # warning percentages
        for warning in ['low_oil_pressure', 'low_fuel_pressure', 'low_battery_voltage','engine_overheating', 'oil_overheating']:
            data[f'perc_{warning}'] = 100 * np.nansum(data[warning]) / data_length

        # driving behavior percentages
        for behavior in ['trail_braking', 'full_throttle', 'coasting_off_throttle','coasting_off_brakes', 'crossing', 'steer_throttle', 'driver_inactive']:
            data[f'perc_{behavior}'] = 100 * np.nansum(data[behavior]) / data_length

        # driving input percentages
        for rate in ['steer_speed_on', 'steer_speed_off', 'brake_speed_on', 'brake_speed_off', 'throttle_speed_on', 'throttle_speed_off']:
            data[f'perc_{rate}'] = 100 * np.nansum(data[rate]) / data_length

        # acceleration/deceleration percentages
        valid_indices = ~np.isnan(data['G_longitudinal'])
        valid_G_long = data['G_longitudinal'][valid_indices]
        if len(valid_G_long) > 0:
            data['perc_acceleration'] = np.sum(valid_G_long > 0) * 100 / len(valid_G_long)
            data['perc_deceleration'] = np.sum(valid_G_long < 0) * 100 / len(valid_G_long)
        else:
            data['perc_acceleration'] = np.nan
            data['perc_deceleration'] = np.nan

        # return all processed data as a DataFrame
        df_processed = pd.DataFrame(data)
        
        # validate that all arrays have the same length
        lengths = {k: len(v) for k, v in data.items() if isinstance(v, (np.ndarray, list))}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Arrays have different lengths: {lengths}")   
        return df_processed
    
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        raise