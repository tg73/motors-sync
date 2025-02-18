# Motors synchronization script
#
# Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

# Import required modules
import os, logging, time, itertools
from datetime import datetime
import numpy as np
from . import z_tilt

# Constants
PLOT_PATH = '~/printer_data/config/adxl_results/motors_sync'  # Path for saving plots
PIN_MIN_TIME = 0.010            # Minimum wait time to enable hardware pin
MOTOR_STALL_TIME = 0.100        # Minimum wait time to enable motor pin
LEVELING_KINEMATICS = (         # Kinematics with interconnected axes
    ['corexy', 'limited_corexy'])

# Dictionary of mathematical models used for motor synchronization
MATH_MODELS = {
    # Polynomial model (linear, quadratic etc)
    "polynomial": lambda fx, coeffs:
        max(np.roots([*coeffs[:-1], coeffs[-1] - fx]).real),
    # Power law model
    "power": lambda fx, coeffs:
        (fx / coeffs[0]) ** (1 / coeffs[1]),
    # Square root model    
    "root": lambda fx, coeffs:
        (fx**2 - 2*coeffs[1]*fx + coeffs[1]**2) / coeffs[0]**2,
    # Hyperbolic model
    "hyperbolic": lambda fx, coeffs:
        coeffs[0] / (fx - coeffs[1]),
    # Exponential model    
    "exponential": lambda fx, coeffs:
        np.log((fx - coeffs[2]) / coeffs[0]) / coeffs[1],
    # Encoder auto model
    "enc_auto": lambda fx, coeffs: (fx / 1e3 / coeffs[0])
}

class AccelHelper:
    """Helper class for accelerometer-based synchronization"""
    
    AXES_LEVEL_DELTA = 2000  # Delta threshold for axes leveling
    ACCEL_FILTER_THRESHOLD = 3000  # Threshold for accelerometer filtering
    
    def __init__(self, axis, chip_name):
        """Initialize accelerometer helper
        
        Args:
            axis: The axis object this helper is for
            chip_name: Name of the accelerometer chip
        """
        self.axis = axis
        self.sync = axis.sync
        self.chip_name = chip_name
        self.chip_type = 'accelerometer'
        self.dim_type = 'magnitude'
        self.config = self.sync.config
        self.printer = self.sync.printer
        self.aclient = None
        self.chip_filter = None
        self.init_chip_config(chip_name)
        axis.calc_deviation = self._calc_magnitude
        axis.detect_move_dir = self._detect_move_dir
        self.gcode = self.printer.lookup_object('gcode')
        self.toolhead = self.printer.lookup_object('toolhead')
        self.reactor = self.printer.get_reactor()

    def init_chip_config(self, chip_name):
        """Initialize accelerometer chip configuration"""
        # Get the accelerometer config object from the printer
        self.accel_config = self.printer.lookup_object(chip_name)

        # Check if accelerometer has a data_rate attribute
        if hasattr(self.accel_config, 'data_rate'):
            # If data rate is high enough, initialize chip filter
            if self.accel_config.data_rate > self.ACCEL_FILTER_THRESHOLD:
                self.axis._init_chip_filter()
            # Otherwise use identity filter that returns data unchanged
            else:
                self.chip_filter = lambda data: data

        # Special case for beacon accelerometer which always has high sample rate
        elif chip_name == 'beacon':
            # Beacon sampling rate > ACCEL_FILTER_THRESHOLD
            self.axis._init_chip_filter()

        # Raise error if accelerometer type is unknown
        else:
            raise self.config.error(f"motors_sync: Unknown accelerometer"
                                    f" '{chip_name}' sampling rate")

    def flush_data(self):
        """Clear accelerometer data"""
        self.aclient.is_finished = False
        self.aclient.msgs.clear()
        self.aclient.request_start_time = None
        self.aclient.request_end_time = None

    def update_start_time(self):
        """Update measurement start time"""
        self.aclient.request_start_time = self.toolhead.get_last_move_time()

    def update_end_time(self):
        """Update measurement end time"""
        self.aclient.request_end_time = self.toolhead.get_last_move_time()

    def start_measurements(self):
        """Start accelerometer measurements"""
        self.aclient = self.accel_config.start_internal_client()

    def finish_measurements(self):
        """Finish accelerometer measurements"""
        self.aclient.finish_measurements()

    def _wait_samples(self):
        """Wait for accelerometer samples to be ready"""
        # Set timeout limit to 5 seconds from now
        lim = self.reactor.monotonic() + 5.
        
        while True:
            # Get current time
            now = self.reactor.monotonic()
            
            # Pause for 10ms to avoid busy waiting
            self.reactor.pause(now + 0.010)
            
            # Check if we have received any messages and have an end time
            if self.aclient.msgs and self.aclient.request_end_time:
                # Get timestamp of last sample from most recent message
                last_mcu_time = self.aclient.msgs[-1]['data'][-1][0]
                
                # If we have samples past the end time, we're done
                if last_mcu_time > self.aclient.request_end_time:
                    return True
                    
                # If we've exceeded the timeout limit, raise error
                elif now > lim:
                    raise self.gcode.error(
                        'motors_sync: No data from accelerometer')

    def _get_accel_samples(self):
        """Get accelerometer samples between start and end time"""
        # Wait for all samples to be collected
        self._wait_samples()

        # Combine all accelerometer messages into one array
        raw_data = np.concatenate(
            [np.array(m['data']) for m in self.aclient.msgs])

        # Find index where samples start after request_start_time
        # side='left' means include the first sample >= start time
        start_idx = np.searchsorted(raw_data[:, 0],
                    self.aclient.request_start_time, side='left')

        # Find index where samples end before request_end_time
        # side='right' means include the last sample <= end time
        end_idx = np.searchsorted(raw_data[:, 0],
                    self.aclient.request_end_time, side='right')

        # Extract samples between start and end indices
        t_accels = raw_data[start_idx:end_idx]

        # Return just the acceleration values (columns 1 onwards)
        # Column 0 contains timestamps which we don't need
        return t_accels[:, 1:]

    def _calc_magnitude(self):
        """Calculate acceleration magnitude
        
        Returns:
            float: Calculated magnitude value
        """
        # Calculate impact magnitude
        vects = self._get_accel_samples()
        vects_len = vects.shape[0]
        # Kalman filter may distort the first values, or in some
        # cases there may be residual values of toolhead inertia.
        # It is better to take a shifted zone from zero.
        static_zone = range(vects_len // 5, vects_len // 3)
        z_cut_zone = vects[static_zone, :]
        z_axis = np.mean(np.abs(z_cut_zone), axis=0).argmax()
        xy_mask = np.arange(vects.shape[1]) != z_axis
        magnitudes = np.linalg.norm(vects[:, xy_mask], axis=1)
        # Add median, Kalman or none filter
        magnitudes = self.chip_filter(magnitudes)
        # Calculate static noise
        static = np.mean(magnitudes[static_zone])
        # Return avg of 5 max magnitudes with deduction static
        magnitude = np.mean(np.sort(magnitudes)[-5:])
        magnitude = np.around(magnitude - static, 2)
        self.axis.update_log(int(magnitude))
        return magnitude

    def _detect_move_dir(self):
        """Detect movement direction by comparing magnitudes
        
        This method determines which direction the motor should move to reduce magnitude:
        1. Start with forward direction (move_dir = 1)
        2. Make a test move and measure the new magnitude
        3. If magnitude increased (got worse), switch to backward direction
        4. If magnitude decreased (improved), keep forward direction
        5. Update the axis state with the chosen direction
        """
        # Initialize direction as forward but unknown
        self.axis.move_dir = [1, 'unknown']
        
        # Make a test move in the forward direction
        self.sync.single_move(self.axis)
        
        # Measure magnitude after the test move
        self.axis.new_magnitude = self.sync.measure(self.axis)
        
        # Update state to show the step was taken
        self.sync.handle_state(self.axis, 'stepped')
        
        # If magnitude increased, we went the wrong way
        # Set direction to backward (-1)
        if self.axis.new_magnitude > self.axis.magnitude:
            self.axis.move_dir = [-1, 'Backward']
        # If magnitude decreased, we went the right way
        # Keep direction as forward (1) 
        else:
            self.axis.move_dir = [1, 'Forward']
            
        # Update state to show direction was determined
        self.sync.handle_state(self.axis, 'direction')
        
        # Save the new magnitude
        self.axis.magnitude = self.axis.new_magnitude


class EncoderHelper:
    """Helper class for encoder-based synchronization"""
    
    AXES_LEVEL_DELTA = 5  # Delta threshold for axes leveling
    MIN_SAMPLE_PERIOD = 0.000400  # Minimum encoder sample period
    
    def __init__(self, axis, chip_name):
        """Initialize encoder helper
        
        Args:
            axis: The axis object this helper is for
            chip_name: Name of the encoder chip
        """
        self.axis = axis
        self.sync = axis.sync
        self.chip_name = 'angle ' + chip_name
        self.chip_type = 'encoder'
        self.dim_type = 'deviation'
        self.config = self.sync.config
        self.printer = self.sync.printer
        self.angle_config = self.printer.lookup_object(self.chip_name)
        self._check_sample_rate()
        self._check_encoder_place()
        self.is_finished = False
        self.samples = []
        self.raw_deviation = 0
        self.request_start_time = None
        self.request_end_time = None
        axis.calc_deviation = self._calc_position
        axis.detect_move_dir = self._detect_move_dir
        self.gcode = self.printer.lookup_object('gcode')
        self.toolhead = self.printer.lookup_object('toolhead')
        self.reactor = self.printer.get_reactor()

    def _check_sample_rate(self):
        """Check if encoder sample rate is high enough"""
        # Get the sample period from the encoder configuration
        per = self.angle_config.sample_period

        # Check if sample period exceeds minimum required period
        # Raise error if sample rate is too low (period too high)
        # This ensures encoder can sample fast enough for accurate measurements
        if per > self.MIN_SAMPLE_PERIOD:
            raise self.config.error(
                f'motors_sync: Encoder sample rate too '
                f'low: {per} < {self.MIN_SAMPLE_PERIOD}')

    def _check_encoder_place(self):
        """Check encoder placement and swap motors if needed"""
        # Get the stepper motor name that the encoder is bound to
        binded_stepper = self.angle_config.calibration.stepper_name
        
        # Get list of stepper motor names for this axis
        axis_steppers = [s.get_name() for s in self.axis.get_steppers()]
        
        # Verify the encoder's bound stepper exists in this axis
        if binded_stepper not in axis_steppers:
            # Raise error if encoder is bound to a stepper not in this axis
            raise self.config.error(
                f"motors_sync: Encoder '{self.chip_name}' stepper: "
                f"'{binded_stepper}' not in '{self.axis.name}' "
                f"axis steppers: '{', '.join(axis_steppers)}'")
                
        # If encoder is bound to second stepper instead of first,
        # swap the steppers so encoder is always on first stepper
        if binded_stepper != axis_steppers[0]:
            self.axis.swap_steppers()

    def handle_batch(self, batch):
        """Handle batch of encoder samples"""
        if self.is_finished:
            return False
        samples = batch['data']
        self.samples.extend(samples)
        return True

    def flush_data(self):
        """Clear encoder data"""
        self.is_finished = False
        self.samples.clear()
        self.request_start_time = None
        self.request_end_time = None

    def update_start_time(self):
        """Update measurement start time"""
        self.request_start_time = self.toolhead.get_last_move_time()

    def update_end_time(self):
        """Update measurement end time"""
        self.request_end_time = self.toolhead.get_last_move_time()

    def start_measurements(self):
        """Start encoder measurements"""
        self.flush_data()
        self.angle_config.add_client(self.handle_batch)

    def finish_measurements(self):
        """Finish encoder measurements"""
        self.toolhead.wait_moves()
        self.is_finished = True

    def _wait_samples(self):
        """Wait for encoder samples to be ready"""
        # Set timeout limit to 5 seconds from now
        lim = self.reactor.monotonic() + 5.
        
        while True:
            # Get current time
            now = self.reactor.monotonic()
            
            # Pause for 10ms to avoid busy waiting
            self.reactor.pause(now + 0.010)
            
            # Check if we have samples and an end time
            if self.samples and self.request_end_time:
                # Get timestamp of most recent sample
                last_mcu_time = self.samples[-1][0]
                
                # If we have samples past the end time, we're done
                if last_mcu_time > self.request_end_time:
                    return True
                    
                # If we've exceeded the timeout limit, raise error
                elif now > lim:
                    raise self.gcode.error(
                        'motors_sync: No data from encoder')

    def _get_encoder_samples(self):
        """Get encoder samples between start and end time"""
        # Wait until we have enough samples
        self._wait_samples()

        # Convert samples list to numpy array for efficient processing
        raw_data = np.array(self.samples)

        # Find index where samples start after request_start_time
        # side='left' includes the first sample >= start time
        start_idx = np.searchsorted(raw_data[:, 0],
                    self.request_start_time, side='left')

        # Find index where samples end after request_end_time
        # side='right' includes all samples <= end time
        end_idx = np.searchsorted(raw_data[:, 0],
                    self.request_end_time, side='right')

        # Extract samples between start and end indices
        t_accels = raw_data[start_idx:end_idx]

        # Return just the encoder position values (column 1)
        # Column 0 contains timestamps which we don't need
        return t_accels[:, 1]

    def normalize_encoder_pos(self, pos):
        """Normalize encoder position to angle and length
        
        Converts raw encoder position into normalized angle and length values:
        1. Calculates angle by dividing 2^16 (65536) by the position value
        2. Calculates length by dividing axis radius by the angle
        
        Args:
            pos: Raw encoder position value
            
        Returns:
            tuple: (angle, length) where:
                angle: Normalized angle in radians
                length: Normalized length based on axis radius
        """
        # Convert position to angle by dividing 2^16 by position
        angle = (1 << 16) / pos
        
        # Calculate normalized length using axis radius divided by angle
        length = self.axis.rd / angle
        
        return angle, length

    def _calc_position(self):
        """Calculate encoder position deviation
        
        This method:
        1. Gets encoder position samples during measurement window
        2. Calculates baseline static position from middle portion of samples
        3. Finds largest deviations from static position
        4. Normalizes deviation to physical units
        5. Updates logs and returns absolute deviation value
        
        Returns:
            float: Absolute position deviation in microns
        """
        # Get array of encoder position samples during measurement
        positions = self._get_encoder_samples()
        
        # Calculate length of position samples array
        poss_len = positions.shape[0]
        
        # Define static zone as middle portion of samples (between 20-33%)
        # This represents baseline position before/after movement
        static_zone = range(poss_len // 5, poss_len // 3)
        
        # Calculate average static/baseline position
        static = np.mean(positions[static_zone])
        
        # Calculate deviations of all positions from static baseline
        deviations = positions - static
        
        # Get indices of 5 largest absolute deviations
        top_dev_ids = np.argsort(np.abs(deviations))[-5:]
        
        # Calculate mean of the 5 largest deviations
        deviation = np.mean(deviations[top_dev_ids])
        
        # Convert raw deviation to normalized angle and length
        dev_norm = self.normalize_encoder_pos(deviation)
        
        # Convert normalized length to microns and round to 2 decimal places
        deviation = np.around(dev_norm[1] * 1e3, 2)
        
        # Get absolute value of deviation
        abs_deviation = abs(deviation)
        
        # Update axis log with integer deviation value
        self.axis.update_log(int(abs_deviation))
        
        # Store raw deviation for direction detection
        self.raw_deviation = deviation
        
        return abs_deviation

    def _detect_move_dir(self):
        """Detect movement direction from raw deviation
        
        This method:
        1. Checks if raw_deviation is negative or positive to determine direction
        2. Sets axis.move_dir with direction multiplier (-1/1) and label
        3. Updates sync state to indicate direction was detected
        4. Sets new_magnitude to current magnitude as baseline
        """
        # Determine movement direction based on sign of raw_deviation
        if self.raw_deviation < 0:
            # Negative deviation means backward movement
            self.axis.move_dir = [-1, 'Backward']  # Set -1 multiplier and 'Backward' label
        else:
            # Positive/zero deviation means forward movement  
            self.axis.move_dir = [1, 'Forward']    # Set 1 multiplier and 'Forward' label

        # Update sync state to indicate direction was detected
        self.sync.handle_state(self.axis, 'direction')

        # Set new_magnitude to current magnitude as baseline for future comparisons
        self.axis.new_magnitude = self.axis.magnitude


class MotionAxis:
    """Class representing a motion axis for synchronization"""
    
    VALID_MSTEPS = [256, 128, 64, 32, 16, 8, 0]  # Valid microstep values
    
    def __init__(self, sync, name, jx):
        """Initialize motion axis
        
        Args:
            sync: The MotorsSync object
            name: Name of the axis (x,y,etc)
            jx: Joint axes configuration
        """
        self.sync = sync
        self.name = name
        self.joint_axes = jx.get(name, [])
        self.config = sync.config
        self.printer = self.config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.move_dir = [1, 'unknown']
        self.move_msteps = 2
        self.actual_msteps = 0
        self.check_msteps = 0
        self.backup_msteps = 0
        self.init_magnitude = 0.
        self.magnitude = 0.
        self.new_magnitude = 0.
        self.curr_retry = 0
        self.is_finished = False
        self.log = []
        stepper = 'stepper_' + name
        st_section = self.config.getsection(stepper)
        min_pos = st_section.getfloat('position_min', 0)
        max_pos = st_section.getfloat('position_max')
        self.rd = st_section.getfloat('rotation_distance')
        fspr = st_section.getint('full_steps_per_rotation', 200)
        self.limits = (min_pos + 10, max_pos - 10, (min_pos + max_pos) / 2)
        self.do_buzz = True
        self.rel_buzz_d = self.rd / fspr * 5
        msteps_dict = {m: m for m in self.VALID_MSTEPS}
        self.microsteps = self.config.getchoice(
            f'microsteps_{name}', msteps_dict, default=0)
        if not self.microsteps:
            self.microsteps = self.config.getchoice(
                'microsteps', msteps_dict, default=16)
        self.move_d = self.rd / fspr / self.microsteps
        sync.add_connect_task(self._init_steppers)
        self._init_chip_helper()
        sync.add_connect_task(self._init_fan)
        self.conf_fan = self.config.get(f'head_fan_{name}', '')
        if not self.conf_fan:
            self.conf_fan = self.config.get('head_fan', None)
        msmax = self.microsteps / 2
        self.max_step_size = self.config.getint(
            f'max_step_size_{name}', default=0, minval=1, maxval=msmax)
        if not self.max_step_size:
            self.max_step_size = self.config.getint(
                'max_step_size', default=3, minval=1, maxval=msmax)
        self.axes_steps_diff = self.config.getint(
            f'axes_steps_diff_{name}', default=0, minval=1)
        if not self.axes_steps_diff:
            self.axes_steps_diff = self.config.getint(
                'axes_steps_diff', self.max_step_size + 1, minval=1)
        rmin = self.move_d * 1e3
        self.retry_tolerance = self.config.getfloat(
            f'retry_tolerance_{name}', default=0, above=rmin)
        if not self.retry_tolerance:
            self.retry_tolerance = self.config.getfloat(
                'retry_tolerance', default=0, above=rmin)
        self.max_retries = self.config.getint(
            f'retries_{name}', default=0, minval=0, maxval=10)
        if not self.max_retries:
            self.max_retries = self.config.getint(
                'retries', default=0, minval=0, maxval=10)

    def flush_motion_data(self):
        """Clear motion data"""
        self.move_dir = [1, 'unknown']
        self.move_msteps = 2
        self.actual_msteps = 0
        self.check_msteps = 0
        self.backup_msteps = 0
        self.init_magnitude = 0.
        self.magnitude = 0.
        self.new_magnitude = 0.
        self.curr_retry = 0
        self.is_finished = False
        self.log = []

    def swap_steppers(self):
        """Swap order of steppers"""
        self.steppers.reverse()

    def get_steppers(self):
        """Get list of steppers"""
        return self.steppers

    def toggle_main_stepper(self, mode, times=None):
        """Toggle main stepper enable/disable
        
        This method controls enabling/disabling of the main stepper motor with timing controls.
        
        Args:
            mode: True to enable motor, False to disable
            times: Tuple of (ontime, offtime) in seconds. If None, uses default stall time.
                  If only one time provided, uses it for ontime and default stall for offtime.
        """
        # Set default timing if none provided - use motor stall time for both on/off
        if times is None:
            times = (MOTOR_STALL_TIME,)*2
        # If only one time provided, use it for ontime and stall time for offtime    
        elif len(times) < 2:
            times = (*times, MOTOR_STALL_TIME)
            
        # Enable/disable the main stepper motor (first stepper) with the specified timing
        self.sync.stepper_enable(self.steppers[0].get_name(), mode, *times)

    def toggle_steppers(self, mode):
        """Toggle all steppers enable/disable
        
        This method enables or disables all stepper motors for this axis with minimal timing.
        
        Args:
            mode: True to enable motors, False to disable
        """
        # Iterate through all stepper motors for this axis
        for st in self.steppers:
            # Enable/disable each stepper with minimal on/off timing
            # PIN_MIN_TIME is used for both enable and disable delays
            # to minimize transition time while still being safe
            self.sync.stepper_enable(st.get_name(), mode, 
                PIN_MIN_TIME, PIN_MIN_TIME)

    def toggle_joint_axes(self, mode):
        """Toggle joint axes steppers enable/disable
        
        This method enables or disables stepper motors for all joint axes.
        Joint axes are axes that move together and need to be synchronized.
        
        Args:
            mode: True to enable motors, False to disable
        """
        # Iterate through all joint axis names
        for name in self.joint_axes:
            # For each joint axis, toggle its stepper motors using the 
            # toggle_steppers() method of that axis's motion controller
            self.sync.motion[name].toggle_steppers(mode)

    def update_log(self, deviation):
        """Update motion log with deviation"""
        self.log.append([int(deviation), self.actual_msteps])

    def _init_steppers(self):
        """Initialize stepper motors
        
        This method:
        1. Gets the printer kinematics object to access stepper configuration
        2. Finds the stepper motors for this axis by name matching
        3. Validates that exactly 2 stepper motors are found
        4. Checks that configured microsteps don't exceed stepper capabilities
        5. Stores the stepper motor objects for later use
        """
        # Get printer kinematics object to access stepper configuration
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        
        # Find stepper motors for this axis by matching names
        # e.g. if self.name is 'x', look for 'stepper_x' in stepper names
        belt_steppers = [s for s in kin.get_steppers()
                         if 'stepper_' + self.name in s.get_name()]
        
        # Validate that exactly 2 stepper motors were found
        # This code only supports dual motor axes
        if len(belt_steppers) not in (2,):
            raise self.config.error(f"motors_sync: Not supported "
                                    f"'{len(belt_steppers)}' count of motors")
        
        # Check each stepper's microstep configuration
        for steppers in belt_steppers:
            # Get the stepper's config section
            st_section = self.config.getsection(steppers.get_name())
            # Get configured microsteps for this stepper
            st_msteps = st_section.getint('microsteps')
            # Validate that our configured microsteps don't exceed stepper's capability
            if self.microsteps > st_msteps:
                raise self.config.error(
                    f'motors_sync: Invalid microsteps count, cannot be '
                    f'more than steppers, {self.microsteps} vs {st_msteps}')
        
        # Store the stepper motor objects for later use
        self.steppers = belt_steppers

    def _init_steps_models(self, def_model):
        """Initialize mathematical models for steps calculation
        
        This method sets up the mathematical models used to calculate motor steps based on measured magnitudes.
        It handles model configuration, validation, and creates a solver function.
        
        Args:
            def_model: Default model parameters to use if not specified for this axis
        
        Raises:
            config.error: If invalid model or coefficients are specified
        """
         # todo: rewrite all func logic
        # Define supported mathematical models and their requirements
        models = {
            'linear': {'ct': 2, 'a': None, 'f': MATH_MODELS['polynomial']},      # Linear model (ax + b)
            'quadratic': {'ct': 3, 'a': None, 'f': MATH_MODELS['polynomial']},   # Quadratic model (ax^2 + bx + c) 
            'power': {'ct': 2, 'a': None, 'f': MATH_MODELS['power']},           # Power model (ax^b)
            'root': {'ct': 2, 'a': 0, 'f': MATH_MODELS['root']},                # Root model (a√x + b)
            'hyperbolic': {'ct': 2, 'a': 0, 'f': MATH_MODELS['hyperbolic']},    # Hyperbolic model (a/x + b)
            'exponential': {'ct': 3, 'a': 0, 'f': MATH_MODELS['exponential']},   # Exponential model (ae^(bx) + c)
            'enc_auto': {'ct': 1, 'a': -1, 'f': MATH_MODELS['enc_auto']},       # Encoder auto model
        }

        # Get model config for this axis, falling back to default if not specified
        model = self.config.getlist(f'steps_model_{self.name}', None)
        if model is None:
            model = self.config.getlist('steps_model', def_model)
            
        # Extract model name and coefficient values
        model_name = model[0]
        coeffs_vals = list(map(float, model[1:]))
        
        # Create coefficient argument names (a, b, c, etc)
        coeffs_args = [chr(97 + i) for i in range(len(coeffs_vals) + 1)]
        
        # Map coefficient names to values
        model_coeffs = {arg: float(val)
                        for arg, val in zip(coeffs_args, coeffs_vals)}
                        
        # Get model configuration and validate
        model_config = models.get(model_name, None)
        if model_config is None:
            raise self.config.error(
                f"motors_sync: Invalid steps model '{model_name}'")
                
        # Validate coefficient count matches model requirements
        if len(model_coeffs) != model_config['ct']:
            raise self.config.error(
                f"motors_sync: Steps model '{model_name}' "
                f"requires {model_config['ct']} coefficients")
                
        # Validate coefficient 'a' meets model constraints
        if model_coeffs['a'] == model_config['a']:
            raise self.config.error(
                f"motors_sync: Coefficient 'a' cannot be "
                f"{model_coeffs['a']} for a '{model_name}' model")
                
        # Store model configuration
        self.model_name = model_name
        self.model_coeffs = tuple(model_coeffs.values())
        
        # Calculate model scale factor based on microsteps
        model_scale = self.microsteps / 16
        if model_name == 'enc_auto':
            model_scale = 1
            
        # Create solver function that applies the model
        def model_solve(fx=None):
            # Use new_magnitude if no input provided
            if fx is None:
                fx = self.new_magnitude
            # Calculate result using model function
            res = model_config['f'](fx, self.model_coeffs)
            # Check for invalid results
            if np.isnan(res):
                self.sync.handle_state(self,
                    f"Microsteps calculation returned NaN "
                    f"for {self.name.capitalize()} axis")
            return res * model_scale
            
        self.model_solve = model_solve

    def _init_chip_filter(self):
        """Initialize chip filter (median/Kalman)
        
        This method sets up filtering for sensor data from either an accelerometer
        or encoder. It supports two filter types:
        - Median filter: Uses a sliding window to calculate median values
        - Kalman filter: Uses a Kalman filter for noise reduction
        
        The filter type and parameters can be configured per-axis or globally.
        """
        # Define valid filter types
        filters = ['default', 'median', 'kalman']
        filters_d = {m: m for m in filters}
        
        # Get filter type from config, first checking axis-specific setting
        filter = self.config.getchoice(f'chip_filter_{self.name}',
                                       filters_d, 'default').lower()
                                       
        # If default, use global filter setting (defaulting to median)
        if filter == 'default':
            filter = self.config.getchoice('chip_filter',
                                           filters_d, 'median').lower()
                                           
        # Configure median filter
        if filter == 'median':
            # Get window size from axis-specific or global config
            window = self.config.getint(f'median_size_{self.name}',
                                        '', minval=3, maxval=9)
            if not window:
                window = self.config.getint('median_size', default=3,
                                            minval=3, maxval=9)
                                            
            # Window size must be odd for median calculation
            if window % 2 == 0: raise self.config.error(
                f"motors_sync: parameter 'median_size' cannot be even")
                
            # Create median filter function using sliding window
            chip_filter = lambda samples, w=window: np.median(
                [samples[i - w:i + w + 1]
                 for i in range(w, len(samples) - w)], axis=1)
                 
        # Configure Kalman filter
        elif filter == 'kalman':
            # Get Kalman coefficients from axis-specific or global config
            coeffs = self.config.getfloatlist(
                f'kalman_coeffs_{self.name}',
                default=tuple('' for _ in range(6)), count=6)
            if not all(coeffs):
                coeffs = self.config.getfloatlist('kalman_coeffs',
                    default=tuple((1.1, 1., 1e-1, 1e-2, .5, 1.)), count=6)
                    
            # Create Kalman filter function
            chip_filter = KalmanLiteFilter(*coeffs).process_samples
            
        # Assign filter to chip_helper if it exists
        if hasattr(self, 'chip_helper'):
            self.chip_helper.chip_filter = chip_filter
        # Otherwise defer assignment until Klippy connects
        else:
            self.sync.add_connect_task(lambda: setattr(
                self.chip_helper, 'chip_filter', chip_filter))

    def _init_chip_helper(self):
        """Initialize accelerometer or encoder helper
        
        This method sets up either an accelerometer or encoder sensor for motor feedback:
        1. Gets axis-specific accelerometer and encoder config names
        2. Validates that only one sensor type is configured
        3. Falls back to global accelerometer config if no axis-specific sensor
        4. Requires at least one sensor type to be configured
        5. For accelerometers:
           - Creates AccelHelper instance on Klippy connect
           - Initializes signal filtering
           - Sets up linear steps model with default params
        6. For encoders:
           - Creates EncoderHelper instance on Klippy connect 
           - Sets up encoder auto steps model
        """
        # Get axis-specific sensor config names
        accel_chip_name = self.config.get(f'accel_chip_{self.name}', '')
        enc_chip_name = self.config.get(f'encoder_chip_{self.name}', '')
        
        # Error if both sensors configured for same axis
        if accel_chip_name and enc_chip_name:
            raise self.config.error(f"motors_sync: Only 1 sensor "
                                    f"type can be selected")
                                    
        # Try global accelerometer config if no axis-specific sensor
        if not accel_chip_name and not enc_chip_name:
            accel_chip_name = self.config.get('accel_chip', '')
            
        # Error if no sensor configured
        if not accel_chip_name and not enc_chip_name:
            raise self.config.error(
                f"motors_sync: Sensors type 'accel_chip' or "
                f"'encoder_chip_<axis>' must be provided")
                
        # Initialize accelerometer
        if accel_chip_name:
            # Create AccelHelper when Klippy connects
            self.sync.add_connect_task(lambda: setattr(self,
                'chip_helper', AccelHelper(self, accel_chip_name)))
            # Set up signal filtering
            self._init_chip_filter()
            # Configure default linear steps model
            def_steps_model = ['linear', 20000, 0]
            self._init_steps_models(def_steps_model)
            
        # Initialize encoder
        elif enc_chip_name:
            # Create EncoderHelper when Klippy connects  
            self.sync.add_connect_task(lambda: setattr(self,
                'chip_helper', EncoderHelper(self, enc_chip_name)))
            # Configure encoder auto steps model
            def_steps_model = ['enc_auto', self.move_d]
            self._init_steps_models(def_steps_model)

    def _create_fan_switch(self, method):
        """Create fan switch function based on fan type
        
        This method creates a fan control function based on the fan type:
        - For heater_fan: Controls fan speed directly 
        - For temperature_fan: Controls fan by setting temperature target
        - For unknown fan type: Creates a no-op function
        
        Args:
            method: String indicating fan type ('heater_fan' or 'temperature_fan')
        """
        # Handle heater_fan type
        if method == 'heater_fan':
            def fan_switch(on=True):
                # Return if no fan configured
                if not self.fan:
                    return
                # Get current time
                now = self.reactor.monotonic()
                # Calculate print time accounting for MCU lag
                print_time = (self.fan.fan.get_mcu().
                              estimated_print_time(now))
                # Set speed to last speed if on, 0 if off
                speed = self.fan.last_speed if on else .0
                # Update fan speed with minimum time delay
                self.fan.fan.set_speed(value=speed,
                    print_time=print_time + PIN_MIN_TIME)
                    
        # Handle temperature_fan type
        elif method == 'temperature_fan':
            def fan_switch(on=True):
                # Return if no fan configured
                if not self.fan:
                    return
                # Store current target temp if not already saved
                if not self.last_fan_target:
                    self.last_fan_target = self.fan.target_temp
                # Set target to saved temp if on, 0 if off    
                target = self.last_fan_target if on else .0
                self.fan.set_temp(target)
            # Initialize saved target temp
            self.last_fan_target = 0
            
        # Handle unknown fan type with no-op function
        else:
            def fan_switch(_):
                return
                
        # Store the created function
        self.fan_switch = fan_switch

    def _init_fan(self):
        """Initialize fan control
        
        This method sets up fan control for motor synchronization:
        1. Defines supported fan types (heater_fan and temperature_fan)
        2. If no fan is configured, creates a no-op stub function
        3. Otherwise tries to find and initialize the configured fan:
           - Attempts to look up the fan object for each supported fan type
           - Creates appropriate fan control function when fan is found
           - Raises error if fan cannot be found or is unsupported type
        """
        # List of supported fan control methods
        fan_methods = ['heater_fan', 'temperature_fan']
        
        # If no fan configured, create stub function and return
        if self.conf_fan is None:
            self._create_fan_switch(None)
            return
            
        # Try to find and initialize the configured fan
        for method in fan_methods:
            try:
                # Look up fan object of current type
                self.fan = self.printer.lookup_object(
                    f'{method} {self.conf_fan}')
                # Create fan control function for this type
                self._create_fan_switch(method)
                return
            except:
                # Try next fan type if lookup fails
                continue
                
        # Raise error if fan not found or unsupported
        raise self.config.error(f"motors_sync: Unknown fan or "
                                f"fan method '{self.conf_fan}'")


class MotorsSync:
    """Main class for motor synchronization"""
    
    def __init__(self, config):
        """Initialize motor synchronization
        
        Args:
            config: Printer configuration object
        """
        self.config = config
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.force_move = self.printer.load_object(config, 'force_move')
        self.stepper_en = self.printer.load_object(config, 'stepper_enable')
        self.printer.register_event_handler("klippy:connect", self._handle_connect)
        self.status = z_tilt.ZAdjustStatus(self.printer)
        self.connect_tasks = []
        # Read config
        self._init_axes()
        self._init_sync_method()
        # Register commands
        self.gcode.register_command('SYNC_MOTORS', self.cmd_SYNC_MOTORS,
                                    desc=self.cmd_SYNC_MOTORS_help)
        self.gcode.register_command('SYNC_MOTORS_CALIBRATE',
                                    self.cmd_SYNC_MOTORS_CALIBRATE,
                                    desc=self.cmd_SYNC_MOTORS_CALIBRATE_help)
        # Variables
        self.reactor = self.printer.get_reactor()
        self._init_stat_manager()

    def add_connect_task(self, task):
        self.connect_tasks.append(task)

    def _handle_connect(self):
        self.toolhead = self.printer.lookup_object('toolhead')
        self.travel_speed = self.toolhead.max_velocity / 2
        self.travel_accel = min(self.toolhead.max_accel, 5000)
        self.kin = self.toolhead.get_kinematics()
        for task in self.connect_tasks: task()
        self.connect_tasks.clear()

    def _check_common_attr(self):
        """Check that certain attributes are consistent across all motion axes
        
        For kinematics with interconnected axes (like CoreXY), certain parameters
        must be identical across all axes for proper synchronization.
        
        Checks the following attributes:
        - microsteps: Number of microsteps per full step
        - model_name: Name of the mathematical model used
        - model_coeffs: Coefficients for the mathematical model
        - max_step_size: Maximum allowed step size
        - axes_steps_diff: Step difference threshold between axes
        
        Raises:
            config.error: If any attributes differ between axes (except chip_name
                         for encoder-based systems)
        """
        # Apply restrictions for LEVELING_KINEMATICS kinematics
        # List of attributes that must be identical across axes
        common_attr = ['microsteps', 'model_name', 'model_coeffs',
                       'max_step_size', 'axes_steps_diff']
        
        # Check each attribute
        for attr in common_attr:
            # Get set of unique values for this attribute across all axes
            diff = set([getattr(cls, attr) for cls in self.motion.values()])
            
            # Skip if all values are identical
            if len(diff) < 2:
                continue
                
            # Special case: Allow different chip names for encoder-based systems
            if (attr == 'chip_name' and list(self.motion.values())[0]
                 .chip_helper.chip_type == 'encoder'):
                continue
                
            # Format error message showing differing values
            params_str = ', '.join(f"'{attr}: {v}'" for v in diff)
            raise self.config.error(
                f"motors_sync: Options {params_str} cannot be "
                f"different for a '{self.conf_kin}' kinematics")

    def _init_axes(self):
        """Initialize printer axes configuration for motor synchronization
        
        This method sets up the axes that will be synchronized based on the printer's
        kinematics type. It handles both CoreXY/similar kinematics (which have 
        interconnected axes that need leveling) and Cartesian kinematics.
        """
        # Define which axes can be synchronized (currently only X and Y supported)
        valid_axes = ['x', 'y']
        
        # Get printer configuration section and kinematics type
        printer_section = self.config.getsection('printer')
        self.conf_kin = printer_section.get('kinematics')
        
        # Handle CoreXY and similar kinematics that need axis leveling
        if self.conf_kin in LEVELING_KINEMATICS:
            self.do_level = True
            # Get X and Y axes from config, defaulting to both if not specified
            axes = [a.lower() for a in self.config.getlist(
                'axes', count=2, default=['x', 'y'])]
            # Define which axes are interconnected (X depends on Y and vice versa)
            joint_ax = {'x': ['y'], 'y': ['x']}
            
        # Handle standard Cartesian kinematics
        elif self.conf_kin == 'cartesian':
            self.do_level = False
            # Get specified axes from config
            axes = [a.lower() for a in self.config.getlist('axes')]
            # No interconnected axes for Cartesian
            joint_ax = {}
            
        # Error if unsupported kinematics type
        else:
            raise self.config.error(f"motors_sync: Not supported "
                                    f"kinematics '{self.conf_kin}'")
                                    
        # Validate that specified axes are supported
        if any(axis not in valid_axes for axis in axes):
            raise self.config.error(f"motors_sync: Invalid axes "
                                    f"parameter '{','.join(axes)}'")
                                    
        # Create motion control objects for each axis
        self.motion = {ax: MotionAxis(self, ax, joint_ax) for ax in axes}
        
        # For CoreXY etc, verify axes have matching parameters
        if self.conf_kin in LEVELING_KINEMATICS:
            self._check_common_attr()

    def _init_sync_method(self):
        """Initialize the synchronization method for motor control
        
        This method determines how multiple motors will be synchronized:
        - sequential: One motor at a time
        - alternately: Switching between motors
        - synchronous: All motors simultaneously 
        - default: Automatically choose based on kinematics
        
        For CoreXY and similar kinematics that have interconnected axes,
        the default is 'alternately'. For standard Cartesian kinematics,
        the default is 'sequential'.
        
        Raises:
            config.error: If an invalid sync method is specified for the kinematics
        """
        # Define valid synchronization methods
        methods = ['sequential', 'alternately', 'synchronous', 'default']
        
        # Get sync method from config, defaulting to 'default'
        self.sync_method = self.config.getchoice(
            'sync_method', {m: m for m in methods}, 'default')
            
        # Handle default sync method based on kinematics type
        if self.sync_method == 'default':
            if self.conf_kin in LEVELING_KINEMATICS:
                self.sync_method = methods[1]  # Use alternately for CoreXY etc
            else:
                self.sync_method = methods[0]  # Use sequential for Cartesian
                
        # Validate sync method is compatible with kinematics
        elif (self.sync_method in methods[1:]  # If not sequential
              and self.conf_kin not in LEVELING_KINEMATICS):  # And not CoreXY
            raise self.config.error(
                f"motors_sync: Invalid sync method: {self.sync_method} "
                f"for '{self.conf_kin}' type kinematics")

    def _init_stat_manager(self):
        """Initialize statistics manager for tracking motor synchronization results
        
        Sets up a statistics manager to track and log:
        - Success rates for each axis
        - Magnitude measurements before/after sync
        - Number of microsteps moved
        - Number of retries needed
        - Min/max magnitudes detected
        - Total sync count
        """
        # Command name for accessing stats
        command = 'SYNC_MOTORS_STATS'
        # CSV file to store stats
        filename = 'sync_stats.csv'
        # CSV column headers
        format = 'axis,status,magnitudes,steps,msteps,retries,date,'

        def log_parser(log):
            """Parse raw log data into formatted statistics
            
            Args:
                log: Raw log data containing sync results
                
            Returns:
                List of formatted statistics strings for each axis
            """
            # Initialize stats dictionary
            a = {}
            out = []
            
            # Process each log entry
            for p in log:
                # Create new axis entry if needed
                a.setdefault(p[0], {
                    'count': 0,                    # Total sync attempts
                    'success': 0,                  # Successful syncs
                    'msteps': 0,                  # Total microsteps moved
                    'magnitudes': [0., 0., 0., 999999.],  # Min/max/start/end magnitudes
                    'retries': 0,                 # Total retries
                })
                
                # Update stats for this axis
                a[p[0]]['count'] += 1
                if p[1]:
                    a[p[0]]['success'] += 1
                # Update start/end magnitudes    
                a[p[0]]['magnitudes'][:2] = (np.add(
                    a[p[0]]['magnitudes'][:2], (p[2][-2], p[2][0])))
                # Track max magnitude
                if p[2].max() > a[p[0]]['magnitudes'][2]:
                    a[p[0]]['magnitudes'][2] = p[2].max()
                # Track min magnitude    
                if p[2].min() < a[p[0]]['magnitudes'][3]:
                    a[p[0]]['magnitudes'][3] = p[2].min()
                # Add microsteps moved    
                a[p[0]]['msteps'] += abs(p[3][-2] / (p[4] / 16))
                # Add retries
                a[p[0]]['retries'] += p[5]

            # Format stats for each axis
            for axis, a in a.items():
                cf_microsteps = self.motion[axis.lower()].microsteps
                st_microsteps = a['msteps'] / a['count'] * (cf_microsteps / 16)
                # Build formatted stats string
                out.append(f"""
                {axis.upper()} axis statistics:
                Successfully synced:     {a['success'] / a['count'] * 100:.2f}%
                Average start magnitude: {a['magnitudes'][1] / a['count']:.2f}
                Average end magnitude:   {a['magnitudes'][0] / a['count']:.2f}
                Average msteps count:    {st_microsteps:.0f}/{cf_microsteps}
                Average retries count:   {a['retries'] / a['count']:.2f}
                Min detected magnitude:  {a['magnitudes'][3]:.2f}
                Max detected magnitude:  {a['magnitudes'][2]:.2f}
                Synchronization count:   {a['count']}
                """)
                out.append('')
            return out

        # Create statistics manager instance
        manager = StatisticsManager(self.gcode, command,
                                    filename, log_parser, format)

        def write_log(axis=None):
            """Write sync results to statistics log
            
            Args:
                axis: Optional specific axis to log, otherwise logs all axes
            """
            if manager.error:
                return
            status = axis is None
            # Get axes to log
            for axis in ([axis] if axis else [
                 a for n, a in self.motion.items() if n in self.axes]):
                if not axis.actual_msteps:
                    continue
                name = axis.name
                magnitudes, pos = zip(*axis.log)
                msteps = axis.microsteps
                retries = axis.curr_retry
                date = datetime.now().strftime('%Y-%m-%d')
                # Write log entry
                manager.write_log([name, status, magnitudes,
                                   pos, msteps, retries, date])
        
        # Store write_log function as instance method
        self.write_log = write_log

    def gsend(self, params):
        self.gcode.run_script_from_command(params)

    def stepper_enable(self, stepper, mode, ontime, offtime):
        """Enable or disable a stepper motor with timing controls
        
        Args:
            stepper: The stepper motor to control
            mode: True to enable motor, False to disable
            ontime: Time to wait before enabling/disabling (in seconds)
            offtime: Time to wait after enabling/disabling (in seconds)
        """
        # Wait for specified time before enabling/disabling
        self.toolhead.dwell(ontime)
        
        # Get current print time for synchronization
        print_time = self.toolhead.get_last_move_time()
        
        # Get the enable line for this stepper motor
        el = self.stepper_en.enable_lines[stepper]
        
        # Enable or disable the motor based on mode
        el.motor_enable(print_time) if mode \
            else el.motor_disable(print_time)
            
        # Wait for specified time after enabling/disabling
        self.toolhead.dwell(offtime)

    def stepper_move(self, mcu_stepper, dist):
        """Move a stepper motor by a specified distance
        
        Args:
            mcu_stepper: The stepper motor object to move
            dist: Distance to move in mm
        
        Uses force_move to directly control the stepper motor at the configured
        travel speed and acceleration, bypassing normal motion planning.
        """
        self.force_move.manual_move(mcu_stepper, dist,
            self.travel_speed, self.travel_accel)

    def single_move(self, axis, mcu_stepper=None, dir=1):
        """Move a stepper motor by a calculated number of microsteps
        
        Args:
            axis: The motor axis object to move
            mcu_stepper: Specific stepper to move (defaults to second stepper)
            dir: Direction multiplier (1 or -1) to control move direction
        """
        # Move <axis>1 stepper motor by default
        # Get the second stepper motor for this axis if none specified
        if mcu_stepper is None:
            mcu_stepper = axis.get_steppers()[1]

        # Calculate number of microsteps to move based on:
        # - axis.move_msteps: Base number of steps to move
        # - axis.move_dir[0]: Direction multiplier from previous moves
        # - dir: Input direction multiplier
        move_msteps = axis.move_msteps * axis.move_dir[0] * dir

        # Convert microsteps to actual distance to move
        dist = axis.move_d * move_msteps

        # Track total microsteps moved for this axis
        axis.actual_msteps += move_msteps
        axis.check_msteps += move_msteps

        # Execute the move using the stepper motor
        self.stepper_move(mcu_stepper, dist)

    def buzz(self, axis, rel_moves=25):
        """Generate fading oscillations on a stepper motor to clear stiction
        
        Args:
            axis: The motor axis object to buzz
            rel_moves: Number of oscillation cycles (default 25)
        """
        # Fading oscillations by <axis>1 stepper
        # Get the second stepper motor for this axis
        mcu_stepper1 = axis.get_steppers()[1]
        
        # Track the last absolute position
        last_abs_pos = 0
        
        # Disable the main stepper motor briefly
        axis.toggle_main_stepper(0, (PIN_MIN_TIME,)*2)
        
        # Generate decreasing amplitude oscillations
        for osc in reversed(range(0, rel_moves)):
            # Calculate absolute position based on relative buzz distance
            # Amplitude decreases as osc counts down
            abs_pos = axis.rel_buzz_d * (osc / rel_moves)
            
            # Move back and forth (+/-) around the position
            for inv in [1, -1]:
                # Invert position
                abs_pos *= inv
                
                # Calculate distance to move from last position
                dist = (abs_pos - last_abs_pos)
                
                # Update last position
                last_abs_pos = abs_pos
                
                # Execute the move
                self.stepper_move(mcu_stepper1, dist)
    def measure(self, axis):
        """Measure the impact/vibration of a motor axis
        
        This method:
        1. Optionally buzzes the motor to clear any stiction
        2. Flushes any existing accelerometer data
        3. Toggles the stepper motor on/off briefly to clear any residual current
        4. Records start time for measurement
        5. Enables the stepper motor
        6. Records end time for measurement
        7. Either buzzes motor again or disables it
        8. Returns calculated deviation/magnitude
        
        Args:
            axis: The motor axis object to measure
            
        Returns:
            float: The calculated deviation/magnitude for this axis
        """
        # Measure the impact
        # Optionally buzz motor first to clear any stiction
        if axis.do_buzz:
            self.buzz(axis)
            
        # Clear any existing accelerometer data
        axis.chip_helper.flush_data()
        
        # Toggle stepper briefly to clear any residual current
        axis.toggle_main_stepper(1, (PIN_MIN_TIME,))
        axis.toggle_main_stepper(0, (PIN_MIN_TIME,))
        
        # Record measurement start time
        axis.chip_helper.update_start_time()
        
        # Enable stepper motor for measurement
        axis.toggle_main_stepper(1)
        
        # Record measurement end time
        axis.chip_helper.update_end_time()
        
        # Either buzz motor again or disable it
        if axis.do_buzz:
            self.buzz(axis, 5)  # Shorter buzz sequence
        else:
            axis.toggle_main_stepper(0)
            
        # Return calculated deviation/magnitude
        return axis.calc_deviation()
    def homing(self):
        """Home axes and move to center position
        
        This method:
        1. Gets current time and extracts axes/configs from motion dictionary
        2. Checks if axes need homing by comparing with kinematics homed status
        3. If needed, sends G28 homing command for the axes
        4. Calculates center position for each axis using their limits
        5. Moves to center position at configured travel speed
        6. Dwells to allow motors to stabilize
        """
        # Homing and going to center
        # Get current time for checking homing status
        now = self.reactor.monotonic()
        
        # Extract axes names and configurations from motion dictionary
        axes, confs = zip(*self.motion.items())
        
        # Check if any axes need homing by comparing with kinematics status
        if ''.join(axes) not in self.kin.get_status(now)['homed_axes']:
            # Send G28 homing command for all axes that need it
            self.gsend(f"G28 {' '.join(axes)}")
            
        # Build center position string using axis limits from configs
        center_pos = ' '.join(f'{a}{c.limits[2]}' for a, c in zip(axes, confs))
        
        # Move to center position at configured travel speed (mm/min)
        self.gsend(f"G0 {center_pos} F{self.travel_speed * 60}")
        
        # Dwell to allow motors to stabilize after movement
        self.toolhead.dwell(MOTOR_STALL_TIME)

    def handle_state(self, axis, state=''):
        """Handle different states during motor synchronization and generate status messages
        
        This method manages the state transitions during motor synchronization, performing
        appropriate actions and generating informative messages for each state.
        
        Args:
            axis: The axis object being synchronized
            state: Current state of synchronization process
                'stepped': After moving motor by some number of microsteps
                'static': After a new magnitude measurement without movement
                'direction': After determining movement direction
                'start': When starting synchronization for an axis
                'done': When synchronization is complete for an axis
                'retry': When retrying synchronization after failure
        """
        name = axis.name.upper()
        dim_type = axis.chip_helper.dim_type

        # After moving motor by some microsteps
        if state == 'stepped':
            msteps = axis.move_msteps * axis.move_dir[0]
            msg = (f"{name}-New {dim_type}: {axis.new_magnitude} "
                   f"on {msteps}/{axis.microsteps} step move")

        # After measuring new magnitude without movement
        elif state == 'static':
            msg = f"{name}-New {dim_type}: {axis.new_magnitude}"

        # After determining movement direction
        elif state == 'direction':
            msg = f"{name}-Movement direction: {axis.move_dir[1]}"

        # When starting synchronization for an axis
        elif state == 'start':
            axis.flush_motion_data()
            axis.fan_switch(False)  # Turn off fan for measurement
            axis.chip_helper.start_measurements()
            axis.init_magnitude = axis.magnitude = self.measure(axis)
            msg = (f"{axis.name.upper()}-Initial {dim_type}: "
                   f"{axis.init_magnitude}")

        # When synchronization is complete for an axis
        elif state == 'done':
            axis.fan_switch(True)
            axis.chip_helper.finish_measurements()
            axis.toggle_main_stepper(1, (PIN_MIN_TIME,)*2)
            msg = (f"{name}-Motors adjusted by {axis.actual_msteps}/"
                   f"{axis.microsteps} step, {dim_type} "
                   f"{axis.init_magnitude} --> {axis.magnitude}")

        # When retrying after synchronization failure
        elif state == 'retry':
            axis.move_dir[1] = 'unknown'  # Reset movement direction
            msg = (f"{name}-Retries: {axis.curr_retry}/{axis.max_retries} "
                   f"Back on last {dim_type}: {axis.magnitude} on "
                   f"{axis.actual_msteps}/{axis.microsteps} step "
                   f"to reach {axis.retry_tolerance}")

        # Handle invalid states by cleaning up and raising error
        else:
            for axis in [c for a, c in self.motion.items() if a in self.axes]:
                axis.fan_switch(True)
                axis.chip_helper.finish_measurements()
            raise self.gcode.error(state)

        # Output status message
        self.gcode.respond_info(msg, True)
    def _axes_level(self, m, s):
        """Level two axes by adjusting their magnitudes to be within a target delta
        
        Args:
            m: Main axis object to adjust
            s: Secondary axis object to compare against
        """
        # Calculate initial magnitude difference between axes
        delta = m.init_magnitude - s.init_magnitude
        target_delta = m.chip_helper.AXES_LEVEL_DELTA
        
        # Return if axes are already within target delta
        if delta <= target_delta:
            return
            
        self.gcode.respond_info(
            f'Start axes level, delta: {delta:.2f}', True)
            
        force_exit = False
        while True:
            # Check if we need to remeasure secondary axis magnitude
            # This happens when steps moved exceeds axes_steps_diff threshold
            steps_diff = abs(abs(m.check_msteps) - abs(s.check_msteps))
            if steps_diff >= m.axes_steps_diff:
                s.new_magnitude = s.magnitude = self.measure(s)
                self.handle_state(s, 'static')
                m.check_msteps, s.check_msteps = 0, 0
                
            # Detect movement direction if unknown
            if m.move_dir[1] == 'unknown':
                m.detect_move_dir()
                
            # Calculate number of steps needed to level axes
            steps_delta = int(m.model_solve() - m.model_solve(s.magnitude))
            m.move_msteps = min(max(steps_delta, 1), m.max_step_size)
            
            # Move main axis and measure new magnitude
            self.single_move(m)
            m.new_magnitude = self.measure(m)
            self.handle_state(m, 'stepped')
            
            # If magnitude got worse (increased)
            if m.new_magnitude > m.magnitude:
                # Move back one step
                self.single_move(m, dir=-1)
                
                # Check if exceeded retry tolerance
                if m.retry_tolerance and m.magnitude > m.retry_tolerance:
                    m.curr_retry += 1
                    if m.curr_retry > m.max_retries:
                        self.handle_state(m, 'done')
                        self.write_log(m)
                        self.handle_state(m, 'Too many retries')
                    self.handle_state(m, 'retry')
                    continue
                force_exit = True
                
            # Update main axis magnitude
            m.magnitude = m.new_magnitude
            
            # Calculate new delta between axes
            delta = m.new_magnitude - s.magnitude
            
            # Exit if axes are leveled or we need to force exit
            if (delta < target_delta
                    or m.new_magnitude < s.magnitude
                    or force_exit):
                self.gcode.respond_info(
                    f"Axes are leveled: {m.name.upper()}: "
                    f"{m.init_magnitude} --> {m.new_magnitude}, "
                    f"{s.name.upper()}: {s.init_magnitude} "
                    f"--> {s.magnitude}, delta: {delta:.2f}", True)
                return
            continue
    def _single_sync(self, m, check_axis=False):
        """Synchronize a single motor axis
        
        Args:
            m: Motor axis object to synchronize
            check_axis: If True, just measure current magnitude without moving
        """
        # "m" is a main axis, just single axis
        # If check_axis is True, just measure current magnitude and return
        if check_axis:
            m.new_magnitude = self.measure(m)
            self.handle_state(m, 'static')
            m.magnitude = m.new_magnitude
            return

        # If movement direction is unknown, determine it
        if m.move_dir[1] == 'unknown':
            # Take initial measurement if needed
            if not m.actual_msteps or m.curr_retry:
                m.new_magnitude = self.measure(m)
                m.magnitude = m.new_magnitude
                self.handle_state(m, 'static')

            # Check if already within tolerance
            if (not m.actual_msteps
                    and m.retry_tolerance
                    and m.new_magnitude < m.retry_tolerance):
                m.is_finished = True
                return

            # Detect which direction reduces magnitude
            m.detect_move_dir()

        # Calculate number of microsteps to move (between 1 and max_step_size)
        m.move_msteps = min(max(
            int(m.model_solve()), 1), m.max_step_size)

        # Move motor and measure new magnitude
        self.single_move(m)
        m.new_magnitude = self.measure(m)
        self.handle_state(m, 'stepped')

        # If magnitude increased (got worse)
        if m.new_magnitude > m.magnitude:
            # Move back one step
            self.single_move(m, dir=-1)

            # Check if exceeded retry tolerance
            if m.retry_tolerance and m.magnitude > m.retry_tolerance:
                m.curr_retry += 1
                if m.curr_retry > m.max_retries:
                    # Log error and mark as failed
                    self.write_log(m)
                    self.handle_state(m, 'done')
                    self.handle_state(m, 'Too many retries')
                self.handle_state(m, 'retry')
                return

            # Mark as finished since we can't improve further
            m.is_finished = True
            return

        # Update magnitude with new improved value
        m.magnitude = m.new_magnitude

    def _run_sync(self):
        """Main synchronization routine that handles different sync methods"""
        # Alternating sync method - synchronize multiple axes one at a time
        if self.sync_method == 'alternately' and len(self.axes) > 1:
            # Find axes with min and max magnitude for leveling
            min_ax, max_ax = [c for c in sorted(
                self.motion.values(), key=lambda i: i.init_magnitude)]
            # Level the axes relative to each other
            self._axes_level(max_ax, min_ax)
            # Order axes based on which has max magnitude
            axes = self.axes[::-1] if max_ax.name == self.axes[0] else self.axes
            # Cycle through axes until all are finished
            for axis in itertools.cycle(axes):
                m = self.motion[axis]
                if m.is_finished:
                    if all(self.motion[ax].is_finished for ax in self.axes):
                        break
                    continue
                self._single_sync(m)

        # Synchronous method - synchronize multiple axes simultaneously
        elif self.sync_method == 'synchronous' and len(self.axes) > 1:
            check_axis = False
            cycling = itertools.cycle(self.axes)
            # Find axis with maximum magnitude
            max_ax = [c for c in sorted(
                self.motion.values(), key=lambda i: i.init_magnitude)][-1]
            max_ax.detect_move_dir()
            while True:
                # Get current and next axis in cycle
                axis = next(cycling)
                cycling, cycle = itertools.tee(cycling)
                m = self.motion[axis]
                sec = next(cycle)
                s = self.motion[sec]
                
                # Check if current axis is done
                if m.is_finished:
                    if all(self.motion[ax].is_finished for ax in self.axes):
                        break
                    continue

                # Check if axes need to be re-measured based on step differences
                if m.magnitude < s.magnitude and not s.is_finished:
                    # None: m['axes_steps_diff'] == s['axes_steps_diff']
                    steps_diff = abs(abs(m.check_msteps) - abs(s.check_msteps))
                    if steps_diff >= m.axes_steps_diff:
                        check_axis = True
                        m.check_msteps, s.check_msteps = 0, 0
                    else:
                        continue
                self._single_sync(m, check_axis)
                check_axis = False

        # Sequential or single axis method
        elif self.sync_method == 'sequential' or len(self.axes) == 1:
            for axis in self.axes:
                m = self.motion[axis]
                # To skip measure() in _single_sync()
                m.detect_move_dir()
                while True:
                    if m.is_finished:
                        if all(self.motion[ax].is_finished for ax in self.axes):
                            return
                        break
                    self._single_sync(m)
        else:
            raise self.gcode.error('Error in sync methods!')

    cmd_SYNC_MOTORS_help = 'Start motors synchronization'
    def cmd_SYNC_MOTORS(self, gcmd, force_run=False):
        """Start motors synchronization process
        
        Args:
            gcmd: G-code command object containing parameters
            force_run: Whether to force sync even if within tolerance
        """
        # Get axes to sync from command parameters
        axes_from_gcmd = gcmd.get('AXES', '')
        if axes_from_gcmd:
            # Parse comma-separated list of axes
            axes_from_gcmd = axes_from_gcmd.split(',')
            # Validate all specified axes exist
            if any([axis not in self.motion.keys()
                    for axis in axes_from_gcmd]):
                raise self.gcode.error(f'Invalid axes parameter')
            self.axes = [axis for axis in axes_from_gcmd]
        else:
            # If no axes specified, use all available axes
            self.axes = list(self.motion.keys())

        # Get accelerometer chip configuration
        chip = gcmd.get(f'ACCEL_CHIP', '')
        for axis in self.axes:
            m = self.motion[axis]
            # Skip non-accelerometer axes
            if m.chip_helper.chip_type != 'accelerometer':
                continue
            
            # Configure accelerometer chip for this axis
            ax_chip = gcmd.get(f'ACCEL_CHIP_{axis.upper()}', chip).lower()
            if ax_chip and ax_chip != m.chip_helper.chip_name:
                try:
                    self.printer.lookup_object(ax_chip)
                except Exception as e:
                    raise self.gcode.error(e)
                self.motion[axis].chip_helper.init_chip_config(ax_chip)

            # Get retry tolerance settings
            retry_tol = gcmd.get_int(f'RETRY_TOLERANCE_{axis.upper()}', 0)
            if not retry_tol:
                retry_tol = gcmd.get_int(f'RETRY_TOLERANCE', 0)
            if retry_tol:
                m.retry_tolerance = retry_tol

            # Get max retries settings  
            retries = gcmd.get_int(f'RETRIES_{axis.upper()}', 0)
            if not retries:
                retries = gcmd.get_int(f'RETRIES', 0)
            if retries:
                m.max_retries = retries

        # Reset status and home axes
        self.status.reset()
        self.homing()
        self.gcode.respond_info('Motors synchronization started', True)

        # Initialize all axes for sync
        for axis in self.axes:
            self.handle_state(self.motion[axis], 'start')

        # Check if all axes are already within tolerance
        if not force_run and all(m.init_magnitude < m.retry_tolerance
             for m in (self.motion[ax] for ax in self.axes)):
            # If within tolerance, finish and report
            retry_tols = ''
            for axis in self.axes:
                m = self.motion[axis]
                m.chip_helper.finish_measurements()
                m.fan_switch(True)
                retry_tols += f'{m.name.upper()}: {m.retry_tolerance}, '
            self.gcode.respond_info(f"Motors magnitudes are in "
                                    f"tolerance: {retry_tols}", True)
        else:
            # Run synchronization if needed
            self._run_sync()
            # Mark axes as done
            for axis in self.axes:
                self.handle_state(self.motion[axis], 'done')

        # Check final results and write log
        self.status.check_retry_result('done')
        self.write_log()

    cmd_SYNC_MOTORS_CALIBRATE_help = 'Calibrate synchronization process model'
    def cmd_SYNC_MOTORS_CALIBRATE(self, gcmd):
        # Calibrate sync model and model coeffs
        if not hasattr(self, 'sync_calibrate_helper'):
            self.sync_calibrate_helper = MotorsSyncCalibrate(self)
        self.sync_calibrate_helper.run_calibrate(gcmd)
        self.status.reset()

    def get_status(self, eventtime):
        return self.status.get_status(eventtime)


class MotorsSyncCalibrate:
    def __init__(self, sync):
        self.sync = sync
        self.gcode = sync.gcode
        try:
            self._load_modules()
        except ImportError as e:
            self.gcode.error(f'Could not import: {e}')
        self.path = os.path.expanduser(PLOT_PATH)
        self.check_export_path()

    @staticmethod
    def _load_modules():
        globals().update({
            'wrap': __import__('textwrap', fromlist=['wrap']).wrap,
            'multiprocessing': __import__('multiprocessing'),
            'plt': __import__('matplotlib.pyplot', fromlist=['']),
            'ticker': __import__('matplotlib.ticker', fromlist=['']),
            'curve_fit': __import__(
                'scipy.optimize', fromlist=['curve_fit']).curve_fit
        })

    def check_export_path(self):
        if os.path.exists(self.path):
            return
        try:
            os.makedirs(self.path)
        except OSError as e:
            raise self.gcode.error(
                f'Error generate path {self.path}: {e}')

    math_models = {
        'linear': (lambda x, a, b: a*x + b, '-.', '#DF8816'),
        'quadratic': (lambda x, a, b, c: a*x**2 + b*x + c, '--', 'green'),
        'power': (lambda x, a, b: a * np.power(x, b), ':', 'cyan'),
        'root': (lambda x, a, b: a * np.sqrt(x) + b, '--', 'magenta'),
        'hyperbolic': (lambda x, a, b: a / x + b, '-.', 'purple'),
        'exponential': (lambda x, a, b, c: a * np.exp(b*x) + c, ':', 'blue')
    }

    def find_best_func(self, x_data, y_data, maxfev=999999999):
        """Find the best mathematical model to fit the calibration data.
        
        Tries fitting multiple mathematical models (linear, quadratic, etc) to the 
        input data and evaluates their accuracy using RMSE (Root Mean Square Error).
        
        Args:
            x_data: Array of x values (microstep positions)
            y_data: Array of y values (measured desynchronization magnitudes)
            maxfev: Maximum number of function evaluations during curve fitting
            
        Returns:
            Tuple containing:
            - List of strings with RMSE and coefficient info for each model
            - Tuple of (x_data, y_data, fitted_functions) for plotting
        """
        # Store results for each mathematical model
        funcs = []
        
        # Try fitting each model to the data
        for name, param in self.math_models.items():
            # Fit the model using scipy's curve_fit
            coeffs, _ = curve_fit(param[0], x_data, y_data, maxfev=maxfev)
            
            # Calculate predicted y values using fitted coefficients
            y_pred = param[0](x_data, *coeffs)
            
            # Calculate RMSE to evaluate model accuracy
            rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
            
            # Store model results
            funcs.append({'name': name, 'rmse': rmse, 'coeffs': coeffs})
            
        # Sort models by RMSE (best fit first)
        funcs = sorted(funcs, key=lambda f: f['rmse'])
        
        # Generate info strings for each model
        info = ['Functions RMSE and coefficients']
        for f in funcs:
            c_str = ','.join([f'{c:.10f}' for c in f['coeffs']])
            info.append(f"{f['name']}: RMSE {f['rmse']:.2f} coeffs: {c_str}")
            
        return info, (x_data, y_data, funcs)
    def plotter(self, x_data, y_data, funcs, axis, accel_chip,
                peak_msteps, fullstep, rmse_lim=20000):
        """Plot calibration data and fitted mathematical models
        
        Args:
            x_data: Array of x values (microstep positions)
            y_data: Array of y values (measured desynchronization magnitudes) 
            funcs: List of fitted function parameters and RMSE values
            axis: The axis being calibrated (X, Y, Z etc)
            accel_chip: Name of accelerometer chip used
            peak_msteps: Maximum microsteps moved during calibration
            fullstep: Full step size used
            rmse_lim: RMSE threshold for filtering models to plot
            
        Returns:
            String with path to saved plot file
        """
        # Set power limits for scientific notation on y-axis
        pow_lim = (-2, 2)
        
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Plot raw calibration data points
        ax.scatter(x_data, y_data, label='Samples',
                   color='red', zorder=2, s=10)
        
        # Generate smooth x values for fitted curves
        x_fit = np.linspace(min(x_data), max(x_data), 200)
        
        # Plot each fitted model curve if RMSE is below threshold
        for func in funcs:
            rmse = func['rmse']
            if rmse < rmse_lim:
                upname = func['name'].capitalize()
                model = self.math_models[func['name']]
                _func = model[0](x_fit, *func['coeffs'])
                c_str = ','.join([f'{c:.3f}' for c in func['coeffs']])
                func_str = f"{upname} RMSE: {rmse:.2f} coeffs: {c_str}"
                linestyle = model[1]
                color = model[2]
                ax.plot(x_fit, _func, label=func_str,
                        linestyle=linestyle, linewidth=1, color=color)
        
        # Configure plot legend
        ax.legend(loc='lower right', fontsize=6, framealpha=1, ncol=1)
        
        # Generate unique filename with timestamp
        accel_chip = accel_chip.replace(' ', '-')
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        lognames = (f'calibrate_plot_{axis}_{peak_msteps}-'
                    f'{fullstep}_{accel_chip}_{now}.png')
        
        # Set plot title and labels
        title = (f"Dependency of desynchronization "
                 f"and functions ({''.join(lognames)})")
        ax.set_title('\n'.join(wrap(title, 66)), fontsize=10)
        ax.set_xlabel(f'Microsteps: 1/{fullstep}')
        
        # Configure x-axis ticks and grid
        ax.set_xticks(np.arange(0, max(x_data) + 2.5, 2.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # Configure y-axis formatting and grid
        ax.set_ylabel('Magnitude')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=pow_lim)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which='major', color='grey')
        ax.grid(which='minor', color='lightgrey')
        
        # Save plot to file
        png_path = os.path.join(self.path, lognames)
        plt.savefig(png_path, dpi=1000)
        
        return f'Access to interactive plot at: {png_path}'

    def save_config(self, axis, func):
        configfile = self.sync.printer.lookup_object("configfile")
        axes = [axis] + [a for a in self.sync.motion[axis].joint_axes]
        for ax in axes:
            pname = f'steps_model_{ax}'
            func_name = func['name']
            func_coeffs = list(map(str, func['coeffs']))
            block = func_name + ',\n  ' + ',\n  '.join(func_coeffs)
            configfile.set('motors_sync', pname, block)
            msg = f"{pname}: {func_name}, {','.join(func_coeffs)}"
            self.gcode.respond_info(msg)
        self.gcode.respond_info(
            f"Motors sync model for '{', '.join(axes)}' axis "
             "has been calibrated.\nThe SAVE_CONFIG command "
             "will update the printer config\n file with new "
             "parameters and restart the printer.")

    def run_calibrate(self, gcmd, fullstep=16):
        """Run motor calibration routine to determine optimal synchronization parameters
        
        Args:
            gcmd: GCode command object containing calibration parameters
            fullstep: Number of microsteps per full step (default 16)
        """
        # Get calibration parameters from GCode command
        repeats = gcmd.get_int('REPEATS', 2, minval=2, maxval=100)  # Number of calibration cycles
        axis = gcmd.get('AXIS').lower()  # Axis to calibrate
        m = self.sync.motion.get(axis, None)  # Get motion object for axis
        if m is None:
            self.gcode.error(f'Invalid axis: {axis.upper()}')
            
        # Get distance to move in microsteps
        peak_mstep = gcmd.get_int('DISTANCE', fullstep,
                                  minval=2, maxval=fullstep*2)
                                  
        # Check if plotting is enabled
        need_plot = False
        need_plot_str = gcmd.get('PLOT', 'True').lower()
        if need_plot_str == 'true' or need_plot_str == '1':
            need_plot = True
            
        # Print calibration parameters
        self.gcode.respond_info(
            f'Calibration started on {axis.upper()} axis with '
            f'{repeats} repeats, move to +-{peak_mstep}/16 microstep')
            
        # Run initial sync before calibration
        self.gcode.respond_info('Synchronizing before calibration...')
        self.sync.cmd_SYNC_MOTORS(gcmd, force_run=True)
        
        # Initialize calibration variables
        max_steps = 0
        invs = [1, -1, -1, 1]  # Direction multipliers for movement pattern
        y_samples = [-1,]  # List to store magnitude samples
        mcu_stepper1 = m.get_steppers()[1]  # Get second stepper motor
        
        # Scale calibration steps based on microstep resolution
        m.move_msteps = m.microsteps // fullstep
        emul_peak_mstep = peak_mstep * m.move_msteps
        looped_pos = itertools.cycle([m.rd, -m.rd, -m.rd, m.rd])  # Cyclic position pattern
        
        # Start calibration state
        self.sync.handle_state(m, 'start')
        
        # Run calibration cycles
        for r in range(1, repeats + 1):
            self.gcode.respond_info(
                f'Repeats: {r}/{repeats} Move to +-'
                f'{emul_peak_mstep}/{m.microsteps} microstep')
                
            # Move stepper to next position in pattern    
            self.sync.stepper_move(mcu_stepper1, next(looped_pos))
            
            # For each direction in pattern
            for inv in invs:
                m.move_dir[0] = inv
                # Take samples at each step
                for _ in range(peak_mstep):
                    self.sync.single_move(m)
                    m.new_magnitude = self.sync.measure(m)
                    self.sync.handle_state(m, 'stepped')
                    if m.new_magnitude > max(y_samples):
                        max_steps += 1
                    y_samples.append(m.new_magnitude)
                    
        # Update final magnitude
        m.magnitude = m.new_magnitude
        
        # Cleanup and finish measurements
        m.fan_switch(True)
        m.chip_helper.finish_measurements()
        
        # Convert samples to numpy arrays and create x values
        y_samples = np.sort(np.array(y_samples[1:]))
        x_samples = np.linspace(0.01, max_steps, len(y_samples))
        
        # Log calibration data
        x_samples_str = ', '.join([f'{i:.2f}' for i in x_samples])
        y_samples_str = ', '.join([str(i) for i in y_samples])
        logging.info(f"motors_sync_calibrate: x = [{x_samples_str}]")
        logging.info(f"motors_sync_calibrate: y = [{y_samples_str}]")
        
        # Find best mathematical model and save to config
        msg, data = self.find_best_func(x_samples, y_samples)
        self.save_config(axis, data[-1][0])
        
        if not need_plot:
            return

        def samples_processing():
            """Generate plot of calibration data in separate process"""
            try:
                os.nice(10)  # Lower process priority
            except:
                pass
            self.gcode.respond_info('Generating a plot...', True)
            msg = self.plotter(*data, axis, m.chip_helper.chip_name,
                               peak_mstep, fullstep)
            self.gcode.respond_info(msg, True)

        # Run plotter in separate process
        proces = multiprocessing.Process(target=samples_processing)
        proces.daemon = False
        proces.start()


class KalmanLiteFilter:
    def __init__(self, A, H, Q, R, P0, x0):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.P = self.st_p = P0
        self.x = self.st_x = x0
        self.I = 1

    def flush_data(self):
        self.x = self.st_x
        self.P = self.st_p

    def predict(self):
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z):
        self.predict()
        y = z - (self.H * self.x)
        S = self.H * self.P * self.H + self.R
        K = self.P * self.H * S
        self.x += K * y
        self.P = (self.I - K * self.H) * self.P
        return self.x

    def process_samples(self, samples):
        self.flush_data()
        return np.array(
            [self.update(z) for z in samples]).reshape(-1)


class StatisticsManager:
    def __init__(self, gcode, cmd_name, log_name, log_parser, format):
        self._load_modules()
        self.gcode = gcode
        self.cmd_name = cmd_name.upper()
        self.log_parser = log_parser
        self.format = format
        # Register commands
        self.gcode.register_command(self.cmd_name, self.cmd_GET_STATS,
                                    desc=self.cmd_GET_STATS_help)
        # Variables
        self.home_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_path = os.path.join(self.home_dir, log_name)
        self.error = ''
        # Checks
        self.check_log()

    @staticmethod
    def _load_modules():
        for module in ['csv', 'ast']:
            globals()[module] = __import__(module)

    def check_log(self):
        if os.path.exists(self.log_path):
            header = ','.join(self.read_log(True))
            if header != self.format:
                self.error = (f'Invalid format, type {self.cmd_name} '
                              f'CLEAR=1 to reset and fix statistics')
        else:
            try:
                self.write_log(self.format.split(','))
            except Exception as e:
                self.error = str(e)

    def read_log(self, only_header=False):
        with open(self.log_path, mode='r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            if only_header:
                return header
            log = list(reader)
        return np.array(log)

    def write_log(self, line):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(line)

    def clear_log(self):
        os.remove(self.log_path)
        self.check_log()
        self.error = ''

    def parse_raw_log(self, log):
        converted = []
        for line in log:
            converted_line = []
            for element in line:
                element = str(element).strip("'\"")
                try:
                    converted_el = ast.literal_eval(element)
                except (ValueError, SyntaxError):
                    converted_el = element
                if isinstance(converted_el, tuple):
                    converted_el = np.array(converted_el)
                converted_line.append(converted_el)
            converted.append(converted_line)
        return np.array(converted, dtype=object)

    cmd_GET_STATS_help = 'Show statistics'
    def cmd_GET_STATS(self, gcmd):
        do_clear = gcmd.get('CLEAR', '').lower()
        if do_clear in ['true', '1']:
            self.clear_log()
            self.gcode.respond_info('Logs was cleared')
            return
        if self.error:
            self.gcode.respond_info(f'Statistics collection is '
                                    f'disabled due:\n{self.error}')
            return
        raw_log = self.read_log()
        if raw_log.size == 0:
            self.gcode.respond_info('Logs are empty')
            return
        log = self.parse_raw_log(raw_log)
        msg = self.log_parser(log)
        for line in msg:
            self.gcode.respond_info(str(line))


def load_config(config):
    return MotorsSync(config)
