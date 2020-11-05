"""This module contains a class for simulating the effect of pile up in HEXITEC."""

import matplotlib.pyplot as plt

import random
import warnings
import math
import timeit
from datetime import datetime
import os
from subprocess import call

import numpy as np
from numpy import ma
import astropy.units as u
from astropy.table import Table, vstack
from astropy.units.quantity import Quantity
from astropy import constants
import pandas as pd
from scipy.special import erf
import scipy.stats as stats

# Defining max number of data points in a pandas dataframe.
DATAFRAME_MAX_POINTS = 1e8

class overall_simulation():
    """
    Runs all aspects of the simulation from start to finish
    This currently only allows you to input photons on a single energy into
    the detector
    """
    def __init__(self, incident_spectrum, incident_xpixel_range=(0,80),incident_ypixel_range=(0,80),readout_xpixel_range=(0,80),\
                 readout_ypixel_range=(0,80), frame_rate=Quantity(3900.,unit=1/u.s), n_photons=100, \
                     photon_rate=Quantity(1000,unit=1/u.s), threshold=None, charge_cloud_sigma=None, \
                         detector_temperature=Quantity(17.2,unit=u.Celsius), bias_voltage=Quantity(500,unit=u.V), \
                             pixel_pitch=Quantity(250,unit=u.um), peak_hold=True, leakage=False, noise=False, \
                                 detector=None, geant_file=None, material="G4_CADMIUM_TELLURIDE",\
                                 detector_thickness=Quantity(1,unit=u.mm)):
    
        self.incident_xpixel_range = incident_xpixel_range
        self.incident_ypixel_range = incident_ypixel_range
        self.readout_xpixel_range = readout_xpixel_range
        self.readout_ypixel_range = readout_ypixel_range
        self.frame_rate = frame_rate
        self.n_photons = n_photons
        self.incident_spectrum = incident_spectrum
        self.threshold = threshold
        self.charge_cloud_sigma = charge_cloud_sigma
        self.detector_temperature = detector_temperature
        self.bias_voltage = bias_voltage
        self.pixel_pitch = pixel_pitch
        self.peak_hold = peak_hold
        self.leakage = leakage
        self.noise = noise
        self.detector = detector
        self.geant_file = geant_file
        self.material = material
        self.detector_thickness = detector_thickness
        self.photon_rate = photon_rate
        
        #setting hits parameter to None - this is overwritten if GEANT4 simulation is performed
        self.hits = None
            
    def run(self):
        """
        Runs overall simulation
        """
        #initialising and running GEANT4 simulations if setting chosen
        if self.geant_file != None:
            geant_sim = Geant_simulations(incident_spectrum=self.incident_spectrum, directory=self.geant_file,incident_xpixel_range=self.incident_xpixel_range,\
                                          incident_ypixel_range=self.incident_ypixel_range, readout_xpixel_range=self.readout_xpixel_range,\
                                              readout_ypixel_range=self.readout_ypixel_range, pixel_pitch=self.pixel_pitch,\
                                                 nphotons=self.n_photons, dthickness = self.detector_thickness,\
                                                      dMat=self.material, photon_rate=self.photon_rate)
            self.hits = geant_sim.run_simulation()
            
        #running python simulation
        simulation = simulate_hexitec_on_spectrum(incident_photons=self.hits,incident_spectrum=self.incident_spectrum, photon_rate=self.photon_rate, n_photons=self.n_photons,\
                                                  frame_rate = self.frame_rate, incident_xpixel_range = self.incident_xpixel_range, \
                                                      incident_ypixel_range=self.incident_ypixel_range, readout_xpixel_range=self.readout_xpixel_range,\
                                                          readout_ypixel_range=self.readout_ypixel_range, threshold=self.threshold,\
                                                              charge_cloud_sigma=self.charge_cloud_sigma, charge_drift_length=self.detector_thickness,\
                                                                  detector_temperature=self.detector_temperature, bias_voltage=self.bias_voltage,\
                                                                      pixel_pitch=self.pixel_pitch, peak_hold=self.peak_hold, leakage=self.leakage, \
                                                                          noise=self.noise, detector=self.detector, geant_file=self.geant_file,detector_thickness = self.detector_thickness)
        
        return simulation


def simulate_hexitec_on_spectrum(
        incident_spectrum, photon_rate, n_photons, incident_photons=None,
        frame_rate=Quantity(3900., unit=1/u.s),
        incident_xpixel_range=(0,80), incident_ypixel_range=(0,80),
        readout_xpixel_range=(0,80), readout_ypixel_range=(0,80),
        threshold=None, charge_cloud_sigma=None,
        charge_drift_length=1*u.mm, detector_temperature=17.2*u.Celsius, bias_voltage=500*u.V, pixel_pitch=250*u.um, peak_hold=True, leakage=True, noise=True, detector=None, geant_file=None, detector_thickness=1*u.mm):
    """
    Simulates how a grid of HEXITEC pixels records photons from a given spectrum.

    Parameters
    ----------
    incident_spectrum : `astropy.table.Table`
        Incident photon spectrum.  Table has following columns:
            lower_bin_edges : `astropy.units.quantity.Quantity`
            upper_bin_edges : `astropy.units.quantity.Quantity`
            counts : array-like
    photon_rate: `astropy.units.Quantity`
        The average rate of photons incident on the region of the detector defined by
        incident_xpixel_range and incident_ypixel_range.  See below.
    n_photons: `int`
        Total number of incident photons.
    frame_rate: `astropy.units.quantity.Quantity`
        Desired frame rate at which HEXITEC operates.
    incident_xpixel_range: `tuple` of length 2
        The min and max in pixel units along the x-axis of the region of the detector
        upon which the incident photons fall.
    incident_ypixel_range: `tuple` of length 2
        The min and max in pixel units along the y-axis of the region of the detector
        upon which the incident photons fall.
    readout_xpixel_range: `tuple` of length 2
        The min and max in pixel units along the x-axis of the region of the detector
        which are read out.
    readout_ypixel_range: `tuple` of length 2
        The min and max in pixel units along the y-axis of the region of the detector
        which are read out.
    threshold: `astropy.units.quantity.Quantity`
        Threshold below which a detection is defined.  Note that voltage pulses are negative.
    charge_cloud_sigma: `astropy.units.quantity.Quantity`
        Standard deviation of the charge cloud induced by a photon detection.
        If this is note set, it is calculated using values of
        charge_drift_length, detector_temperature and bias_voltage, defined below.
    charge_drift_length: `astropy.units.quantity.Quantity`
        Vertical distance through detector over which charge drifts once it is absorbed.
    detector_temperature: `astropy.units.quantity.Quantity`
        Temperature of detector.
    bias_voltage: `astropy.units.quantity.Quantity`
        Bias voltage across detector.
    peak_hold: `bool`
        Determines voltage response of detector. If peak_hold is True,
        signal decays following peak response. If peak_hold is False, 
        integrating behaviour is seen with no decay following peak
        response.
        Default = True
    leakage: `bool`
        Determines whether leakage current is accounted for in voltage
        response. If leakage is True, leakage is accounted for.
        Default = True
    noise: `bool`
        Determines whether random electronics noise is accounted for in
        voltage response. If noise is True, noise is included.
        Default = True
    integration_frames: `int`
        Number of frames between charge resets if peak_hold is set to
        True.
        Default = 100
    detector: `str`
        Allows user to load in preset detector parameters.

    Returns
    -------
    hs: HexitecSimulation
        Class holding results and parameters of the simulation.
        hs.measured_spectrum: `astropy.table.Table`
            Measured spectrum taking photon masking of HEXITEC pixel into account.
        hs.measured_photons: `astropy.table.Table`
            See description of "self.measured_photons" in "Returns" section of
            docstring of generate_random_photons_from_spectrum().
        hs.incident_photons : `astropy.table.Table`
            Table of photons incident on the pixel.  Contains the following columns
            time: Amount of time passed from beginning of observing until photon hit.
            energy: Energy of each photon.
        hs.incident_spectrum: `astropy.table.Table`
          Same as input incident_spectrum.

    """
    if incident_photons == None:
        # Generate random photons incident on detector.
        incident_photons = Random_photon_generation().generate_random_photons(incident_spectrum=incident_spectrum, photon_rate=photon_rate, n_photons=n_photons,incident_xpixel_range=incident_xpixel_range, incident_ypixel_range=incident_ypixel_range)
    # Simulate how HEXITEC records incident photons.
    hs = simulate_hexitec_on_photon_list(
        incident_photons, frame_rate=frame_rate,
        incident_xpixel_range=incident_xpixel_range, incident_ypixel_range=incident_ypixel_range,
        readout_xpixel_range=readout_xpixel_range, readout_ypixel_range=readout_ypixel_range,
        threshold=threshold, charge_cloud_sigma=charge_cloud_sigma,
        charge_drift_length=charge_drift_length, detector_temperature=detector_temperature,
        bias_voltage=bias_voltage, peak_hold=peak_hold, leakage=leakage, noise=noise, detector=detector, detector_thickness=detector_thickness)
    hs.incident_spectrum = incident_spectrum
    return hs

class Geant_simulations():
    """
    Class used to initialise, run and analyse GEANT4 simulations
    """
    def __init__(self, incident_spectrum, directory=None, incident_xpixel_range=(0,80),\
                 incident_ypixel_range=(0,80), readout_xpixel_range=(1,79),\
                     readout_ypixel_range=(1,79), pixel_pitch=Quantity(250,unit=u.um),\
                         nphotons=100, dthickness=Quantity(1,unit=u.mm), dMat="G4_CADMIUM_TELLURIDE",\
                                 photon_rate = Quantity(1000,unit=1/u.s)):
        
        #loading in input parameters
        self.directory = directory
        self.incident_xpixel_range = incident_xpixel_range
        self.incident_ypixel_range = incident_ypixel_range
        self.readout_xpixel_range = readout_xpixel_range
        self.readout_ypixel_range = readout_ypixel_range
        self.pixel_pitch = pixel_pitch
        self.incident_spectrum = incident_spectrum
        self.nphotons = nphotons
        self.dthickness = dthickness
        self.dMat = dMat
        self.photon_rate = photon_rate
        
        #setting template directory
        #self.ref_directory = 'C:/Users/uhf41485/Documents/Simulation/Geant4/One_detector_test2' 
        self.ref_directory = '/datadisk/uhf41485/basic'
        self.template_directory = self.ref_directory + '/One_detector_test_source'
        
        #saving inputed parameters for later readout
        args = locals()
        self.parameters = {k: args[k] for k in args.keys() - {'self'}}
    
    def run_simulation(self):
        """
        Runs GEANT4 simulation based on the parameters set upon class 
        initialisation.
        """
        #printing out parameters set for the simulation to the user
        print("Parameters set are:", self.parameters)
        
        #obtaining detector dimensions and photon incident range based on
        #readout and incident detector ROIs
        self.dx, self.dy, self.inc_x, self.inc_y = \
            self.ROI_to_detector_beam_parameters(self.pixel_pitch, self.incident_xpixel_range,\
                                                 self.incident_ypixel_range, self.readout_xpixel_range,\
                                                     self.readout_ypixel_range)
        
        #editing detector parameters in source files
        self.configure_detector_files()
        #editing photon production parameters in source files
        #self.configure_beam_files()
        #editing incident spectrum
        self.configure_spectrum_files()
        #editing bash file used to run GEANT4 simulation
        self.configure_executable()
        
        #run executable and wait for it to finish
        call([self.ref_directory + "/RunSimulation"])
        
        #obtain hits data and save figure
        hits = self.photon_list_from_geant(self.photon_rate)
        self.generate_figures()
        
        return hits
        
    def _pixel_dimen_to_SI_dimen(self,pixel_pitch,inc_x_range,\
                                  inc_y_range,read_x_range,\
                                      read_y_range):
         """
         Converts pixel ROI (e.g. (0,80)) into a SI length - this is used
         to configure beam and directory parameters within GEANT4 source
         files.
         """
         #calculating incident and readout lengths in pixel dimensions
         inc_x_len_pix = inc_x_range[1] - inc_x_range[0]
         inc_y_len_pix = inc_y_range[1] - inc_y_range[0]
         read_x_len_pix = read_x_range[1] - read_x_range[0]
         read_y_len_pix = read_y_range[1] - read_y_range[0]
         
         #converting incident and readout lengths into um
         inc_x_len_si = inc_x_len_pix * pixel_pitch.to(u.mm)
         inc_y_len_si = inc_y_len_pix * pixel_pitch.to(u.mm)
         read_x_len_si = read_x_len_pix * pixel_pitch.to(u.mm)
         read_y_len_si = read_y_len_pix * pixel_pitch.to(u.mm)
        
         return inc_x_len_si, inc_y_len_si, read_x_len_si, read_y_len_si
    
    def ROI_to_detector_beam_parameters(self, pixel_pitch, inc_x_range, \
                                         inc_y_range, read_x_range, \
                                             read_y_range):
        """
        Determines incident ROI of photon beam and size of detector GEANT4
        object based on ROI set.
        """
        inc_x_len_si, inc_y_len_si, read_x_len_si, read_y_len_si \
            = self._pixel_dimen_to_SI_dimen(pixel_pitch, inc_x_range,\
                                            inc_y_range, read_x_range,\
                                                read_y_range)
       
        dx = read_x_len_si * 0.5
        dy = read_y_len_si * 0.5
        inc_x = inc_x_len_si
        inc_y = inc_y_len_si
       
        return dx,dy,inc_x,inc_y
     
    def configure_detector_files(self):
        """
        Edits GEANT4 source files corresponding to the detector's
        configuration.
        """
        for file in os.listdir(self.template_directory + "/src"):
            if file.endswith("DetectorConstruction.cc"):
                filepath =  self.template_directory + "/src/" + file
        
        with open(filepath,'r') as file:
            content = file.readlines()
        
        #changing detector material
        if self.dMat != None:
            dmat_str = "G4Material* det_mat ="
            dmat_ln = [ind for ind,ln in enumerate(content) if dmat_str in ln]
            if len(dmat_ln) == 0:
                print("Could not locate det_mat attribute in " + filepath)
            content[dmat_ln[0]] = 'G4Material* det_mat = nist->FindOrBuildMaterial("' + self.dMat + '"); \n'
         
        #changing detector thickness
        if self.dthickness != None:
            dthick_str = "G4double dthick ="
            dthick_ln = [ind for ind,ln in enumerate(content) if dthick_str in ln]
            if len(dthick_ln) == 0:
                print("Could not locate dthick attribute in " + filepath)
            content[dthick_ln[0]] = "G4double dthick = " + str(self.dthickness.to(u.mm).value) + "*mm; \n"
            
        #changing x and y dimensions of detector
        if ((self.dx != None) or (self.dy != None)):
            dx_str = "G4double dx ="
            dx_ln = [ind for ind,ln in enumerate(content) if dx_str in ln]
            if len(dx_ln) == 0:
                print("Could not locate dx attribute in " + filepath)
            if self.dx != None:
                content[dx_ln[0]] = "G4double dx = " + str(self.dx.to(u.mm).value) + "*mm; \n"
            if self.dy != None:
                content[dx_ln[0]+1] = "G4double dy = " + str(self.dy.to(u.mm).value) + "*mm; \n"

        with open(filepath,'w') as file:
            file.writelines(content)
            
    def configure_beam_files(self):
        """
        Edits GEANT4 source files corresponding to the photon beam's 
        configuration
        """
        filepath = self.template_directory + "/run1.mac"
        
        if not os.path.exists(filepath):
            print("No run file exists. Ending here")
            return
        
        with open(filepath,'r') as file:
            content = file.readlines()

        #changing the location of the incident x-ray beam
        if ((self.inc_x != None) or (self.inc_y != None)):
            beam_pos_str = "/control/alias beam_x "
            beam_pos_ln = [ind for ind,ln in enumerate(content) if beam_pos_str in ln]
            if len(beam_pos_ln) == 0:
                print("Could not locate beam_x attribute in " + filepath)
            if self.inc_x != None:
                content[beam_pos_ln[0]] = "/control/alias beam_x " + str(self.inc_x.to(u.mm).value*0.5) + "\n" 
            if self.inc_y != None:
                content[beam_pos_ln[0]+1] = "/control/alias beam_y " + str(self.inc_y.to(u.mm).value*0.5) + "\n"

        #changing the number of photons
        if self.nphotons != None:
            nphoton_str = "/run/beamOn "
            nphoton_ln = [ind for ind,ln in enumerate(content) if nphoton_str in ln]
            if len(nphoton_ln) == 0:
                print("Could not locate /run/beamOn command in " + filepath)
            content[nphoton_ln[0]] = "/run/beamOn " + str(self.nphotons) + "\n"
        
        with open(filepath,'w') as file:
            file.writelines(content) 
    
    def configure_spectrum_files(self):
        """
        Edits spectrum.csv file based o
        n inputed incident astropy spectrum
        """
        file = self.ref_directory + '/spectrum.txt'
        
        spectrum = np.zeros((len(self.incident_spectrum),2))
        spectrum[:,0] = self.incident_spectrum['upper_bin_edges'].to(u.MeV).value
        spectrum[:,1] = self.incident_spectrum['counts']
        
        #saving file - values rounded to 3 sig figs
        np.savetxt(file,spectrum,fmt="%.3g")
        
    def configure_executable(self):
        """
        Edits bash executable for running GEANT4 simulations
        """
        filepath = self.ref_directory + "/RunSimulation"
        
        with open(filepath,'r') as file:
            content = file.readlines()
        
        dir_str = "mkdir "
        dir_ln = [ind for ind,ln in enumerate(content) if dir_str in ln]
        
        if len(dir_ln) == 0:
            print("Could not locate mkdir command in " + filepath)
        content[dir_ln[0]] = "mkdir " + self.directory + "\n"
        content[dir_ln[0]+1] = "cd " + self.directory + "\n"
        
        with open(filepath,'w') as file:
            file.writelines(content)
        
    def photon_list_from_geant(self, photon_rate):
        """
        Retrieves photon_list from .csv file produced by GEANT4 and converts x and
        y dimensions into pixel dimensions
        """
        
        file = self.ref_directory + '/' + self.directory + '/Analysis_nt_Hits.csv'
        
        #retrieving data from csv file
        data = pd.read_csv(file,delimiter=',',skiprows=13,names=['EventID','TrackID','Time','x','y','z','Energy','Type','ParID'])
        
        #calculating the number of gamma hits within readout. Gamma hits are used as the basis for Python input
        gammas = np.where(data['Type'].values=='gamma')[0]
        
        #creating empty pd dataframe
        input_data = pd.DataFrame(index=np.arange(len(gammas)),columns=['Time','x','y','z','Energy'])
        
        #calculating the indexes and number of unique events 
        events = np.unique(data['EventID']).astype(int)
        n_photons = len(events)
        #generating rough hit times for each event based on poisson distribution
        event_times= Random_photon_generation()._generate_random_photon_times(photon_rate, n_photons)
        
        #calculating time for each gamma hit as sum of event time and global time outputted by GEANT4
        input_data['Time'] = (data['Time'][gammas].values*1e-9) + np.squeeze(event_times[[np.where(events==data['EventID'][ind])[0] for ind in gammas]].value)
        
        #calculating energy for each gamma hit
        #also calculating the weighted x,y,z position based on energies summed into single hit
        for i in range(len(gammas)-1):
            input_data['Energy'].iloc[i] = sum(data['Energy'][gammas[i]:gammas[i+1]].values)
            input_data['x'].iloc[i] = sum((data['x'][gammas[i]:gammas[i+1]].values)*(data['Energy'][gammas[i]:gammas[i+1]].values))/sum(data['Energy'][gammas[i]:gammas[i+1]].values)
            input_data['y'].iloc[i] = sum((data['y'][gammas[i]:gammas[i+1]].values)*(data['Energy'][gammas[i]:gammas[i+1]].values))/sum(data['Energy'][gammas[i]:gammas[i+1]].values)
            input_data['z'].iloc[i] = sum((data['z'][gammas[i]:gammas[i+1]].values)*(data['Energy'][gammas[i]:gammas[i+1]].values))/sum(data['Energy'][gammas[i]:gammas[i+1]].values)
        #calculating last values outside of loop to prevent indexing error
        input_data['Energy'].iloc[-1] = sum(data['Energy'][gammas[-1]:].values)
        input_data['x'].iloc[-1] = sum((data['x'][gammas[-1]:].values)*(data['Energy'][gammas[-1]:].values))/sum(data['Energy'][gammas[-1]:].values)
        input_data['y'].iloc[-1] = sum((data['y'][gammas[-1]:].values)*(data['Energy'][gammas[-1]:].values))/sum(data['Energy'][gammas[-1]:].values)
        input_data['z'].iloc[-1] = sum((data['z'][gammas[-1]:].values)*(data['Energy'][gammas[-1]:].values))/sum(data['Energy'][gammas[-1]:].values)
        
        #converting x and y parameters into pixel dimensions
        input_data['x'] = input_data['x']/self.pixel_pitch.to(u.mm).value + self.readout_xpixel_range[0]
        input_data['y'] = input_data['y']/self.pixel_pitch.to(u.mm).value + self.readout_ypixel_range[0]
        
        return Table([input_data['Time'].values*u.s, input_data['Energy'].values*1000*u.keV, input_data['x'].values, input_data['y'].values, input_data['z'].values*u.mm], names=("time", "energy", "x", "y","z"))

        
    def generate_figures(self):
        """
        Generates figures relating to incident photons and hits at the detector
        """
    
        hit_file = self.ref_directory + '/' + self.directory + '/Analysis_nt_Hits.csv'
        photon_file = self.ref_directory + '/' + self.directory + '/Analysis_nt_Photons.csv'
        
        #retrieving photon and hit data from csv files
        hit_data = np.genfromtxt(hit_file, delimiter=',')
        photon_data = np.genfromtxt(photon_file, delimiter=',')
        
        #creating histogram based on incident photon energies
        photon_energy = photon_data[:,3]*1000
        photon_ehist_fig_loc = self.ref_directory + '/' + self.directory + '/Photon_energy_hist.png'
        photon_ehist_fig = plt.figure()
        plt.hist(photon_energy,bins=photon_energy.max().astype(int),range=(0,photon_energy.max()))
        plt.xlim(left=0)
        plt.title('Energies of incident photons')
        plt.xlabel('Energy [$\it{keV}$]')
        plt.ylabel('Counts')
        plt.savefig(photon_ehist_fig_loc)
        
        #creating histogram based on energy data
        energy = hit_data[:,6]*1000
        ehist_fig_loc = self.ref_directory + '/' + self.directory + '/Energy_hist.png'
        ehist_fig = plt.figure()
        plt.hist(energy,bins=(np.ceil(energy.max()-energy.min()).astype(int)))
        plt.xlim(left=0)
        plt.title('Energy deposited at the detector')
        plt.xlabel('Energy [$\it{keV}$]')
        plt.ylabel('Counts')
        plt.savefig(ehist_fig_loc)
        
        #creating histogram based on depth of interaction
        depth = hit_data[:,5]
        dhist_fig_loc = self.ref_directory + '/' + self.directory + '/Interaction_depth_hist.png'
        dhist_fig = plt.figure()
        plt.hist(depth,bins=100,range=(0,self.dthickness.to(u.mm).value))
        plt.xlim(left=0)
        plt.title('Depth of interaction')
        plt.xlabel('Interaction depth [$\it{mm}$]')
        plt.ylabel('Counts')
        plt.savefig(dhist_fig_loc) 
        
        #creating histogram based on number of hits per primary photon
        events = np.unique(hit_data[:,0]).astype(int)
        num_hits = np.zeros(len(events))
        for ind,ev in enumerate(events):
            num_hits[ind] = len(np.where(hit_data[:,0]==ev)[0])
        nhitshist_fig_loc = self.ref_directory + '/' + self.directory + '/Num_hits_per_event_hist.png'
        nhitshist_fig = plt.figure()
        plt.hist(num_hits,bins=num_hits.max().astype(int))
        plt.xlim(left=0)
        plt.title('Number of energy deposit events per primary photon')
        plt.xlabel('Number of events')
        plt.ylabel('Counts')
        plt.savefig(nhitshist_fig_loc)
        
        
        print("")
        print("Finished producing GEANT4 figures at {0}".format(datetime.now()))
        print("")
        
class Random_photon_generation():
    """
    """
    def __init__(self):
        """
        Initialises Random_photon_generation class.
        """
    def generate_random_photons(self, incident_spectrum, photon_rate, n_photons, incident_xpixel_range=(0,80), incident_ypixel_range=(0,80)):
        
        """
        Generates a set number of incident photons.
        """
        photon_times = self._generate_random_photon_times(photon_rate, n_photons)
        #generate photon energies.
        photon_energies = self._generate_random_photon_energies_from_spectrum(
            incident_spectrum, photon_rate, n_photons)
        #generate photon hit locations.
        x_locations, y_locations = self._generate_random_photon_locations(
            incident_xpixel_range, incident_ypixel_range, n_photons)
        #for random photon generation, assume depth of interaction is 0
        z_locations = np.zeros(len(n_photons))*u.mm
        # Associate photon times, energies and locations.
        return Table([photon_times, photon_energies, x_locations, y_locations,z_locations],
                     names=("time", "energy", "x", "y","z"))
        
    def _generate_random_photon_times(self, photon_rate, n_photons):
        """Generates random photon times."""
        # Generate random waiting times before each photon.
        photon_waiting_times = Quantity(
            np.random.exponential((1./photon_rate).to(u.s).value, n_photons), unit='s')
        return photon_waiting_times.cumsum()
    
    def _generate_random_photon_energies_from_spectrum(self,incident_spectrum,\
                                                       photon_rate, n_photons):
        """
        Converts an input photon spectrum to a probability distribution.
    
        Parameters
        ----------
        incident_spectrum : `astropy.table.Table`
            Incident photon spectrum.  Table has following columns:
                lower_bin_edges : `astropy.units.quantity.Quantity`
                upper_bin_edges : `astropy.units.quantity.Quantity`
                counts : array-like
          photon_rate : `astropy.units.quantity.Quantity`
              Average rate at which photons hit the pixel.
          n_photons : `int`
              Total number of random counts to be generated.
    
        Returns
        -------
        photon_energies : `astropy.units.quantity.Quantity`
            Photon energies
    
        """
        if type(photon_rate) is not Quantity:
            raise TypeError("photon_rate must be an astropy.units.quantity.Quantity")
        n_counts = int(n_photons)
        # Calculate cumulative density function of spectrum for lower and
        # upper edges of spectral bins.
        cdf_upper = np.cumsum(incident_spectrum["counts"])
        cdf_lower = np.insert(cdf_upper, 0, 0.)
        cdf_lower = np.delete(cdf_lower, -1)
        # Generate random numbers representing CDF values.
        print("Generating random numbers for photon energy transformation.")
        time1 = timeit.default_timer()
        randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        # Generate array of spectrum bin indices.
        print("Transforming random numbers into photon energies.")
        time1 = timeit.default_timer()
        bin_indices = np.arange(len(incident_spectrum["lower_bin_edges"]))
        # Generate random energies from randomly generated CDF values.
        photon_energies = Quantity([incident_spectrum["lower_bin_edges"].data[
            bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
            for r in randoms], unit=incident_spectrum["lower_bin_edges"].unit)
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        return photon_energies
    
    def _generate_random_photon_locations(self, incident_xpixel_range,\
                                          incident_ypixel_range, n_photons):
        """Generates random photon hit locations."""
        # Generate random x locations for each photon.
        x = np.random.uniform(incident_xpixel_range[0], incident_xpixel_range[1], n_photons)
        y = np.random.uniform(incident_ypixel_range[0], incident_ypixel_range[1], n_photons)
        return x, y
        

def simulate_hexitec_on_photon_list(
        incident_photons, frame_rate=Quantity(3900., unit=1/u.s),
        incident_xpixel_range=(0,80), incident_ypixel_range=(0,80),
        readout_xpixel_range=(0,80), readout_ypixel_range=(0,80),
        threshold=None, charge_cloud_sigma=None,
        charge_drift_length=1*u.mm, detector_temperature=17.2*u.Celsius, bias_voltage=500*u.V, peak_hold=True, leakage=True, noise=True, detector=None, detector_thickness=1*u.mm):
    """
    Simulates how HEXITEC pixels record photons from a given photon list.

    Parameters
    ----------
    incident_photons: `astropy.table.Table`
        Incident photon spectrum.  Table has following columns:
            time: `astropy.units.quantity.Quantity`
                Time between current photon hit and previous photon hit.
                For first photon in list, this number is time of photon hit
                since beginning of simulation, i.e. since time=0.
            energy: `astropy.units.quantity.Quantity`
                Energy of photon.
            x: `float`
                Position along x-axis in pixel units of photon detection.
            y: `float`
                Position along y-axis in pixel units of photon detection.
    frame_rate: `astropy.units.quantity.Quantity`
        Desired frame rate at which HEXITEC operates.
    incident_xpixel_range: `tuple` of length 2
        The min and max in pixel units along the x-axis of the region of the detector
        upon which the incident photons fall.
    incident_ypixel_range: `tuple` of length 2
        The min and max in pixel units along the y-axis of the region of the detector
        upon which the incident photons fall.
    readout_xpixel_range: `tuple` of length 2
        The min and max in pixel units along the x-axis of the region of the detector
        which are read out.
    readout_ypixel_range: `tuple` of length 2
        The min and max in pixel units along the y-axis of the region of the detector
        which are read out.
    threshold: `astropy.units.quantity.Quantity`
        Threshold below which a detection is defined.  Note that voltage pulses are negative.
    charge_cloud_sigma: `astropy.units.quantity.Quantity`
        Standard deviation of the charge cloud induced by a photon detection.
        If this is note set, it is calculated using values of
        charge_drift_length, detector_temperature and bias_voltage, defined below.
    charge_drift_length: `astropy.units.quantity.Quantity`
        Vertical distance through detector over which charge drifts once it is absorbed.
    detector_temperature: `astropy.units.quantity.Quantity`
        Temperature of detector.
    bias_voltage: `astropy.units.quantity.Quantity`
       Bias voltage across detector.
    peak_hold: `bool`
        Determines voltage response of detector. If peak_hold is True,
        signal decays following peak response. If peak_hold is False, 
        integrating behaviour is seen with no decay following peak
        response.
        Default = True
    leakage: `bool`
        Determines whether leakage current is accounted for in voltage
        response. If leakage is True, leakage is accounted for.
        Default = True
    noise: `bool`
        Determines whether random electronics noise is accounted for in
        voltage response. If noise is True, noise is included.
        Default = True
    integration_frames: `int`
        Number of frames between charge resets if peak_hold is set to
        True.
        Default = 100
    detector: `str`
        Allows user to load in preset detector parameters.

    Returns
    -------
    hs: HexitecSimulation
        Class holding results and parameters of the simulation.
        hs.measured_spectrum : `astropy.table.Table`
          Measured spectrum taking photon masking of HEXITEC pixel into account.
        hs.measured_photons : `astropy.table.Table`
          See description of "self.measured_photons" in "Returns" section of
          docstring of generate_random_photons_from_spectrum().
        hs.incident_photons : `astropy.table.Table`
          Table of photons incident on the pixel.  Contains the following columns
          time : Amount of time passed from beginning of observing
            until photon hit.
          energy : Energy of each photon.

    """
    # Define simulation instance
    hs = HexitecSimulation(
        frame_rate=frame_rate,
        incident_xpixel_range=incident_xpixel_range, incident_ypixel_range=incident_ypixel_range,
        readout_xpixel_range=readout_xpixel_range, readout_ypixel_range=readout_ypixel_range,
        charge_cloud_sigma=charge_cloud_sigma, threshold=threshold,
        charge_drift_length=charge_drift_length, detector_temperature=detector_temperature,
        bias_voltage=bias_voltage, peak_hold=peak_hold, leakage=leakage, noise=noise,
        detector=detector,detector_thickness=detector_thickness)
    #calculating charge cloud sigmas for incident photons
    incident_photons = hs.generate_charge_sigma_photonlist(incident_photons=incident_photons)
    hs.incident_photons = incident_photons
    # Produce photon list accounting for charge sharing.
    pixelated_photons = hs.account_for_charge_sharing_in_photon_list(
        incident_photons, hs._n_1d_neighbours)
    # Separate photons by pixel and simulate each pixel's
    # measurements.
    measured_photons = Table([Quantity([], unit=hs.incident_photons["time"].unit),
                              Quantity([], unit=hs.incident_photons["energy"].unit),
                              [], []], names=("time", "energy", "x", "y"))
    for j in range(hs.readout_ypixel_range[0], hs.readout_ypixel_range[1]):
        for i in range(hs.readout_xpixel_range[0], hs.readout_xpixel_range[1]):
            print("Processing photons hitting pixel ({0}, {1}) of {2} at {3}".format(
                    i, j, (hs.readout_xpixel_range[1]-1,
                           hs.readout_ypixel_range[1]-1), datetime.now()))
            time1 = timeit.default_timer()
            # Find which photons in list, if any, are in given pixel.
            w = np.logical_and(pixelated_photons["x_pixel"] == i,
                               pixelated_photons["y_pixel"] == j)
            # If photons did hit given pixel, simulate how HEXITEC
            # interprets them.
            if w.any():
                # Define threshold for pixel.
                if hs.threshold.shape == ():
                    threshold = hs.threshold
                else:
                    threshold = hs.threshold[i-hs.readout_xpixel_range[0],
                                               j-hs.readout_ypixel_range[0]]
                pixel_measured_photons = \
                  hs.simulate_hexitec_on_photon_list_1pixel(pixelated_photons[w],
                                                              threshold=threshold)
                # Add pixel info to pixel_measured_photons table.
                pixel_measured_photons["x"] = [i]*len(pixel_measured_photons)
                pixel_measured_photons["y"] = [j]*len(pixel_measured_photons)
                measured_photons = vstack((measured_photons, pixel_measured_photons))
            time2 = timeit.default_timer()
            print("Finished processing pixel ({0}, {1}) of {2} in {3} s.".format(
                i, j, (hs.readout_xpixel_range[1]-1, hs.readout_ypixel_range[1]-1),
                time2-time1))
            print(" ")
    # Sort photons by time and return to object.
    measured_photons.sort("time")
    hs.measured_photons = measured_photons

    return hs


class HexitecSimulation():
    """Simulates how HEXITEC records incident photons."""

    def __init__(self, frame_rate=Quantity(3900., unit=1/u.s),
                 incident_xpixel_range=(0,80), incident_ypixel_range=(0,80),
                 readout_xpixel_range=(0,80), readout_ypixel_range=(0,80),
                 charge_cloud_sigma=None, charge_drift_length=1*u.mm,
                 detector_temperature=None, bias_voltage=None, threshold=None,\
                     peak_hold=True, leakage=True, noise=True,\
                         integration_frames=100, detector=None, detector_thickness=1*u.mm):
        """
        Instantiates a HexitecPileUp object.

        Parameters
        ----------
        frame_rate : `astropy.units.quantity.Quantity`
            Operating frame rate of ASIC.
        incident_xpixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction upon which incident photons fall.
        incident_ypixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            y-direction upon which incident photons fall.
        readout_xpixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction to be read out.
        readout_ypixel_range : 2-element `tuple`
            The lower and upper edges of the range of pixels in the
            x-direction to be read out.
        charge_cloud_sigma : `astropy.units.quantity.Quantity`
            Standard deviation of charge cloud assuming it to be a
            2D symmetric gaussian. Default=None.
            If not set, the charge clous standard deviation is calculated
            with self._charge_cloud_sigma() using charge_drift_length,
            detector_temperature, and bias_voltage inputs (below).
        charge_drift_length : `astropy.units.quantity.Quantity`
            Drift length of charge cloud from site of photon interaction to
            anode. Given by CdTe thickness - mean free path of photon in CdTe.
            Default=1mm
        detector_temperature : `astropy.units.quantity.Quantity`
            Operating temperature of the detector.
            Default=None
        bias_voltage : `astropy.units.quantity.Quantity`
            Operating bias voltage of detector.
            Default=None
        threshold: `astropy.units.quantity.Quantity`
            Threshold(s) below which photons are not recorded.  Can be a
            single value or have same shape as readout pixel region,
            (number_readout_xpixels, number_readout_ypixels), one threshold
            value for each readout pixel.  If single value, that value is
            applied to all pixels.  If array, the threshold value in each
            element is applied to the corresponding pixel.
            Must be in units of energy or voltage.  If unit is energy,
            threshold refers to photon energy.  If unit is voltage, threshold
            refers to voltage induced in pixel due to a photon hit which is a
            function of the photon's energy.
            Default=None implies a threshold of 0V
        peak_hold: `bool`
            Determines voltage response of detector. If peak_hold is True,
            signal decays following peak response. If peak_hold is False, 
            integrating behaviour is seen with no decay following peak
            response.
            Default = True
        leakage: `bool`
            Determines whether leakage current is accounted for in voltage
            response. If leakage is True, leakage is accounted for.
            Default = True
        noise: `bool`
            Determines whether random electronics noise is accounted for in
            voltage response. If noise is True, noise is included.
            Default = True
        integration_frames: `int`
            Number of frames between charge resets if peak_hold is set to
            True.
            Default = 100
        detector: `str`
            Allows user to load in preset detector parameters.
        """
        # Define some magic numbers. N.B. _sample_unit must be in string
        # format so it can be used for Quantities and numpy datetime64.
        self._sample_unit = 'ns'
        self.ramp_step = Quantity(1., unit=u.us)
        self._sample_step = Quantity(100., unit=self._sample_unit)
        self._voltage_peaking_time = Quantity(2., unit='us').to(self._sample_unit)
        self._voltage_decay_time = Quantity(8., unit='us').to(self._sample_unit)
        self.peak_hold = peak_hold
        self.leakage = leakage
        self.noise = noise
        self.noise_fwhm = Quantity(0.8, unit=u.keV)
        self.leakage = Quantity(300., unit = u.nA/(u.mm**2))
        self.capacitor = Quantity(15., unit = u.pF)
        self._reset_window = Quantity(100, unit = u.ns)
        self._w_factor = Quantity(4.46, unit=u.eV)
        self._voltage_pulse_shape = self._define_voltage_pulse_shape()
        # Set frame duration from inverse of input frame_rate and round
        # to nearest multiple of self._sample_step.
        self.frame_duration = Quantity(
            round((1./frame_rate).to(self._sample_unit).value/self._sample_step.value
                  )*self._sample_step).to(1/frame_rate.unit)
        self.incident_xpixel_range = incident_xpixel_range
        self.incident_ypixel_range = incident_ypixel_range
        self.readout_xpixel_range = readout_xpixel_range
        self.readout_ypixel_range = readout_ypixel_range
        self._n_1d_neighbours = 3
        self.pixel_pitch = 250*u.um
        self.detector_thickness = detector_thickness
        # Define threshold
        if threshold is None:
            threshold = 0.*u.V
        # Check threshold is of correct type and shape
        if type(threshold) is not Quantity:
            raise TypeError("threshold must be an astropy.units.quantity.Quantity")
        if threshold.shape != () and threshold.shape != \
                (readout_xpixel_range[1]-readout_xpixel_range[0],
                 readout_ypixel_range[1]-readout_ypixel_range[0]):
            raise TypeError("threshold must be a single value or have the same shape as the "
                            "same shape as the readout pixel region, "
                            "i.e. {0}".format((readout_xpixel_range[1]-readout_xpixel_range[0],
                                               readout_ypixel_range[1]-readout_ypixel_range[0])))
        self.threshold = threshold
        self.integration_frame = integration_frames
        # Changing detector parameters if input if detector input is used
        if detector != None:
            self._detector_parameters_call(detector)
            
        
    def simulate_hexitec_on_spectrum_1pixel(self, incident_spectrum, photon_rate, n_photons):
        """
        Simulates how a single HEXITEC pixel records photons from a given spectrum.

        This simulation is a 1st order approximation of the effect of pile up.
        It assumes than only the most energetic photon incident on the detector
        within the period of a single frame is recorded.

        Parameters
        ----------
        incident_spectrum : `astropy.table.Table`
          Incident photon spectrum.  Table has following columns:
            lower_bin_edges : `astropy.units.quantity.Quantity`
            upper_bin_edges : `astropy.units.quantity.Quantity`
            counts : array-like

        photon_rate : `astropy.units.quantity.Quantity`
          The average photon rate.

        n_photons : `int`
          Number of counts to simulate hitting the detector.  Note that the first
          count will not be included in output spectrum as there is no information
          on the waiting time before it.

        Returns
        -------
        self.measured_spectrum : `astropy.table.Table`
          Measured spectrum taking photon masking of HEXITEC pixel into account.
        self.measured_photons : `astropy.table.Table`
          See description of "self.measured_photons" in "Returns" section of
          docstring of generate_random_photons_from_spectrum().
        self.incident_photons : `astropy.table.Table`
          See description of "photons" in "Returns" section of docstring of
          generate_random_photons_from_spectrum().
        self.incident_spectrum : `astropy.table.Table`
          Same as input incident_spectrum.
        self.photon_rate : `astropy.units.quantity.Quantity`
          Same as input photon_rate.

        """
        # Generate random photon energies from incident spectrum to
        # enter detector.
        incident_photons = self.generate_random_photons_from_spectrum(
            incident_spectrum, photon_rate, n_photons)
        # Mark photons which were recorded and unrecorded using a
        # masked array.  Result recorded in self.measured_photons.
        self.measured_photons = self.simulate_hexitec_on_photon_list_1pixel(incident_photons)
        # Convert measured photon list into counts into bins with same
        # bins as the incident spectrum.  N.B. Measured photons can
        # have energies outside incident spectrum energy range.  For
        # these bins, use mean bin width of incident spectrum.
        # N.B. The exact values of incident spectrum bins must be used
        # as rounding errors/approximations can cause erroneous
        # behaviour when binning counts.
        bin_width = np.mean(
            incident_spectrum["upper_bin_edges"]-incident_spectrum["lower_bin_edges"])
        lower_bins = np.arange(incident_spectrum["lower_bin_edges"][0],
                               measured_photons["energy"].min()-bin_width,
                               -bin_width).sort()
        upper_bins = np.arange(incident_spectrum["upper_bin_edges"][-1]+bin_width,
                               measured_photons["energy"].max()+bin_width, bin_width)
        bin_edges = np.concatenate(
            (lower_bins[:-1], hpu.incident_spectrum["lower_bin_edges"],
             np.array([hpu.incident_spectrum["upper_bin_edges"][-1]]), upper_bins))
        # Return an astropy table of the measured spectrum.
        measured_counts = np.histogram(self.measured_photons["energy"], bins=bin_edges)[0]
        self.measured_spectrum = Table(
            [self.incident_spectrum["lower_bin_edges"],
             self.incident_spectrum["upper_bin_edges"], measured_counts],
            names=("lower_bin_edges", "upper_bin_edges", "counts"))


    def generate_random_photons_from_spectrum(self, incident_spectrum, photon_rate, n_photons):
        """Converts an input photon spectrum to a probability distribution.

        Parameters
        ----------
        incident_spectrum : `astropy.table.Table`
          Incident photon spectrum.  Table has following columns:
            lower_bin_edges : `astropy.units.quantity.Quantity`
            upper_bin_edges : `astropy.units.quantity.Quantity`
            counts : array-like
        photon_rate : `astropy.units.quantity.Quantity`
          Average rate at which photons hit the pixel.
        n_photons : `int`
          Total number of random counts to be generated.

        Returns
        -------
        photons : `astropy.table.Table`
          Table of photons incident on the pixel.  Contains the following columns
          time : Amount of time passed from beginning of observing
            until photon hit.
          energy : Energy of each photon.
        self.incident_spectrum : `astropy.table.Table`
          Same as input incident_spectrum.
        self.photon_rate : `astropy.units.quantity.Quantity`
          Same as input photon_rate.

        """
        self.incident_spectrum = incident_spectrum
        if type(photon_rate) is not Quantity:
            raise TypeError("photon_rate must be an astropy.units.quantity.Quantity")
        self.photon_rate = photon_rate
        n_counts = int(n_photons)
        # Calculate cumulative density function of spectrum for lower and
        # upper edges of spectral bins.
        cdf_upper = np.cumsum(self.incident_spectrum["counts"])
        cdf_lower = np.insert(cdf_upper, 0, 0.)
        cdf_lower = np.delete(cdf_lower, -1)
        # Generate random numbers representing CDF values.
        print("Generating random numbers for photon energy transformation.")
        time1 = timeit.default_timer()
        randoms = np.asarray([random.random() for i in range(n_counts)])*cdf_upper[-1]
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        # Generate array of spectrum bin indices.
        print("Transforming random numbers into photon energies.")
        time1 = timeit.default_timer()
        bin_indices = np.arange(len(self.incident_spectrum["lower_bin_edges"]))
        # Generate random energies from randomly generated CDF values.
        photon_energies = Quantity([self.incident_spectrum["lower_bin_edges"].data[
            bin_indices[np.logical_and(r >= cdf_lower, r < cdf_upper)][0]]
            for r in randoms], unit=self.incident_spectrum["lower_bin_edges"].unit)
        # Generate random waiting times before each photon.
        photon_waiting_times = Quantity(
            np.random.exponential(1./self.photon_rate.value, n_photons), unit='s')
        # Associate photon energies and time since start of
        # observation (time=0) in output table.
        photons = Table([photon_waiting_times.cumsum(), photon_energies],
                        names=("time", "energy"))
        time2 = timeit.default_timer()
        print("Finished in {0} s.".format(time2-time1))
        return photons

    def account_for_charge_sharing_in_photon_list(self, incident_photons, n_1d_neighbours):
        """
        Divides photon hits among neighbouring pixels by the charge sharing process.

        """
        # For each photon create extra pseudo-photons in nearest
        # neighbours due to charge sharing.
        n_neighbours = n_1d_neighbours**2
        n_photons_shared = len(incident_photons)*n_neighbours
        times = np.full(n_photons_shared, np.nan)
        x_pixels = np.full(n_photons_shared, np.nan)
        y_pixels = np.full(n_photons_shared, np.nan)
        energy = np.full(n_photons_shared, np.nan)
        neighbor_positions = np.array([""]*n_photons_shared, dtype="S10")
        for i, photon in enumerate(incident_photons):
            # Find fraction of energy in central & neighbouring pixels.
            x_shared_pixels, y_shared_pixels, fractional_energy_in_pixels, \
            pixel_neighbor_positions = self._divide_charge_among_pixels(
                    photon["x"], photon["y"], photon['sigma_x'],
                    photon['sigma_y'], n_1d_neighbours)
            # Insert new shared photon parameters into relevant list.
            times[i*n_neighbours:(i+1)*n_neighbours] = photon["time"]
            x_pixels[i*n_neighbours:(i+1)*n_neighbours] = x_shared_pixels
            y_pixels[i*n_neighbours:(i+1)*n_neighbours] = y_shared_pixels
            energy[i*n_neighbours:(i+1)*n_neighbours] = \
              photon["energy"]*fractional_energy_in_pixels
            neighbor_positions[i*n_neighbours:(i+1)*n_neighbours] = pixel_neighbor_positions
        # Discard any charge lost at edges of detector and events with
        # 0 energy.
        w = np.logical_and(
                np.logical_and(x_pixels >= self.readout_xpixel_range[0],
                               x_pixels < self.readout_xpixel_range[1]),
                np.logical_and(y_pixels >= self.readout_ypixel_range[0],
                               y_pixels < self.readout_ypixel_range[1]),
                energy > 0.)
        # Combine shared photons into new table.
        pixelated_photons = Table([Quantity(times[w], incident_photons["time"].unit),
                                   Quantity(energy[w], incident_photons["energy"].unit),
                                   x_pixels[w], y_pixels[w], neighbor_positions[w]],
                                  names=("time", "energy", "x_pixel", "y_pixel",
                                         "neighbor_positions"))
        return pixelated_photons


    def _divide_charge_among_pixels(self, x, y, x_sigma, y_sigma, n_1d_neighbours):
        """Divides charge-shared photon hits into separate photon hits."""
        # Generate pixel numbers of central pixel & nearest neighbours.
        x_hit_pixel = int(x)
        y_hit_pixel = int(y)
        half_nearest = (n_1d_neighbours-1)/2
        neighbours_range = np.arange(-half_nearest, half_nearest+1)
        x_shared_pixels = np.array([x_hit_pixel+i for i in neighbours_range]*n_1d_neighbours)
        y_shared_pixels = np.array(
            [[y_hit_pixel+i]*n_1d_neighbours for i in neighbours_range]).flatten()
        neighbor_positions = ["down left", "down", "down right", "left", "central",
                              "right", "up left", "up", "up right"]
        # Find fraction of charge in each pixel.
        return x_shared_pixels, y_shared_pixels, self._integrate_gaussian2d(
            (x_shared_pixels, x_shared_pixels + 1), (y_shared_pixels, y_shared_pixels + 1),
            x, y, x_sigma, y_sigma), neighbor_positions

    def generate_charge_sigma_photonlist(self, incident_photons):
        """
        Generates charge cloud dimensions for list of incident photons
        """
        #creating empty astropy table columns for x and y charge cloud sigma
        incident_photons['sigma_x'] = np.zeros(len(incident_photons))
        incident_photons['sigma_y'] = np.zeros(len(incident_photons))
        
        #if user inputted charge cloud sigma, use this value
        if self.charge_cloud_sigma != None:
            incident_photons['sigma_x']=incident_photons['sigma_y'] = self.charge_cloud_sigma
        
        else:
            #calculating charge cloud sigma in SI units
            for i in range(len(incident_photons)):
                charge_drift_length = self.detector_thickness - incident_photons['z'][i]
                incident_photons['sigma_x'][i]=incident_photons['sigma_y'][i] = self._charge_cloud_sigma(charge_drift_length, self.detector_temperature, self.bias_voltage)
    
        #converting charge cloud sigma into pixel units
        incident_photons['sigma_x'] = incident_photons['sigma_x'].to(u.um).value/self.pixel_pitch.to(u.um).value
        incident_photons['sigma_y'] = incident_photons['sigma_y'].to(u.um).value/self.pixel_pitch.to(u.um).value   
        
        return incident_photons
    def _charge_cloud_sigma(self, charge_drift_length, detector_temperature, bias_voltage):
        """
        Returns the standard deviation of the charge cloud when is reaches the anode.

        Parameters
        ----------
        charge_drift_length : `astropy.units.quantity.Quantity`
            Drift length of charge cloud from site of photon interactiont to anode.
            Given by CdTe thickness - mean free path of photon in CdTe.
        detector_temperature : `astropy.units.quantity.Quantity`
            Operating temperature of the detector.
        bias_voltage : `astropy.units.quantity.Quantity`
            Operating bias voltage of detector.

        Returns
        -------
        sigma : `astropy.units.quantity.Quantity`
            1D standard deviation of charge cloud at anode.

        References
        ----------
        [1] : Veale et al. (2014), Measurements of Charege Sharing in Small Pixelated Detectors
        [2] : Iniewski et al. (2007)

        """
        # Determine initial radius of charge cloud (FWHM), r0, from
        # empirical relation derived by Veale et al. (2014) (Fig. 8).
        r0 = Quantity(
            0.1477*detector_temperature.to(u.Celsius, equivalencies=u.temperature()).value+14.66,
            unit="um")
        # Determine radius (FWHM) at anode.
        r = r0 + 1.15*charge_drift_length*np.sqrt(
            2*constants.k_B.si.value*detector_temperature.to(
                u.K, equivalencies=u.temperature()).value/ \
                (constants.e.si.value*abs(bias_voltage.si.value)))
        # Convert FWHM to sigma.
        return r.to(u.um)/1.15


    def _integrate_gaussian(self, limits, mu, sigma):
        return (1/(sigma*np.sqrt(2*np.pi))) * np.sqrt(np.pi/2)*sigma* \
          (erf((limits[1]-mu)/(np.sqrt(2)*sigma))-erf((limits[0]-mu)/(np.sqrt(2)*sigma)))


    def _integrate_gaussian2d(self, x_limits, y_limits, x_mu, y_mu, x_sigma, y_sigma):
        return self._integrate_gaussian(x_limits, x_mu, x_sigma)* \
          self._integrate_gaussian(y_limits, y_mu, y_sigma)


    def simulate_hexitec_on_photon_list_1pixel(self, incident_photons, threshold=None):
        """
        Simulates how HEXITEC records incoming photons in a single pixel.

        Given a list of photons entering the pixel, this function simulates the
        voltage vs. time timeseries in the HEXITEC ASIC caused by the photon hits.
        From that, it then determines the measured photon list.  This simulation
        includes effects "masking" and "pulse pile up".  It assumes that the photon
        waiting times are distributed exponentially with an average rate of
        photon_rate.

        Parameters
        ----------
        incident_photons : `astropy.table.Table`
          Table of photons incident on the pixel.  Contains the following columns
          time : Amount of time passed from beginning of observing until photon hit.
          energy : Energy of each photon.

        threshold: `astropy.units.quantity.Quantity`
            Threshold below which photons are not recorded.
            Must be in units of energy or voltage.  If unit is energy,
            threshold refers to photon energy.  If unit is voltage, threshold
            refers to voltage induced in pixel due to a photon hit which is a
            function of the photon's energy.
            Default=None implies a threshold of 0V


        Returns
        -------
        self.measured_photons : `astropy.table.Table`
          Photon list as measured by HEXITEC.  Table format same as incident_photons.
        self.incident_photons : `astropy.table.Table`
          Same as input incident_photons.

        """
        #self.incident_photons = incident_photons
        sample_step = self._sample_step.to(incident_photons["time"].unit)
        frame_duration = self.frame_duration.to(incident_photons["time"].unit)
        samples_per_frame = int(round((frame_duration/sample_step.to(frame_duration.unit)).value))
        frame_duration_in_sample_unit = int(round(frame_duration.to(self._sample_unit).value))
        # Determine how many frames will be needed to model all photons
        # hitting pixel.
        # Get frame number from time 0 of all photons in pixel.
        photon_frame_numbers = np.array(incident_photons["time"]/frame_duration.value).astype(int)
        # Extend frame numbers to include adjacent frames to allow for
        # frame carryover while ensuring no duplicate frames.
        if self.peak_hold == True:
            extended_frame_numbers = np.unique(np.array(
                [[x-1, x, x+1] for x in photon_frame_numbers]).ravel())
        # If self.peak_hold is False, extending frame numbers to cover the 
        # whole of integration period and first frame following the
        # reset
        if self.peak_hold == False:               
            extended_frame_numbers = np.array([],dtype=int)
            for x in photon_frame_numbers:
                integration_frame_in = x % self.integration_frame
                integration_frame_left = (self.integration_frame-((x+1)%self.integration_frame)) + 2
                extended_frame_numbers = np.append(extended_frame_numbers,[x-1,x,x+1])
                extended_frame_numbers = np.append(extended_frame_numbers,[[x-i] for i in range(integration_frame_in)])
                extended_frame_numbers = np.append(extended_frame_numbers,[[x+i] for i in range(integration_frame_left)])
        extended_frame_numbers = np.unique(extended_frame_numbers.ravel()).astype(int)
        extended_frame_numbers = extended_frame_numbers[np.where(extended_frame_numbers >= 0)[0]]
        # Determine total frames required for all photons hitting pixel.
        total_n_frames = len(extended_frame_numbers)
        # Break incident photons into sub time series manageable for a
        # pandas time series.
        # Determine max frames per subseries.
        subseries_max_frames = int(DATAFRAME_MAX_POINTS*sample_step.value/frame_duration.value)
        subseries_frame_edges = extended_frame_numbers[np.arange(0, total_n_frames,
                                                                 subseries_max_frames)]
        if subseries_frame_edges[-1] < extended_frame_numbers[-1]:
            subseries_frame_edges = np.append(subseries_frame_edges, extended_frame_numbers[-1])
        # Determine number of subseries
        n_subseries = len(subseries_frame_edges)-1
        # Define arrays to hold measured photons
        measured_photon_times = np.array([], dtype=float)
        measured_photon_energies = np.array([], dtype=float)
        next_frame_first_energies = np.array([], dtype=float)
        measured_subframe_photon_times = np.array([], dtype=float)
        # Use for loop to analyse each sub timeseries.
        print("Photons will be analysed in {0} sub-timeseries.".format(n_subseries))
        for i in range(n_subseries):
            print("Processing subseries {0} of {1} at {2}".format(i+1, n_subseries,
                                                                  datetime.now()))
            time1 = timeit.default_timer()

            # Determine which photons are in current subseries.  Include
            # photons in the frames either side of the subseries edges
            # as their voltage pulses may influence the first and last
            # frames of the subseries.  Any detections in these outer
            # frames should be removed later.
            subseries_first_frame = subseries_frame_edges[i]-1
            subseries_last_frame = subseries_frame_edges[i+1]+1
            subseries_incident_photons = incident_photons[np.logical_and(
                photon_frame_numbers >= subseries_first_frame,
                photon_frame_numbers < subseries_last_frame)]

            # Model voltage signal inside HEXITEC ASIC in response to
            # photon hits.
            # Define some basic parameters of subseries.
            subseries_frame_numbers = extended_frame_numbers[np.logical_and(
                extended_frame_numbers >= subseries_first_frame,
                extended_frame_numbers < subseries_last_frame)]
            n_subseries_frames = len(subseries_frame_numbers)
            subseries_photon_frame_numbers = photon_frame_numbers[np.logical_and(
                photon_frame_numbers >= subseries_first_frame,
                photon_frame_numbers < subseries_last_frame)]
            # Generate time stamps for sparse timeseries & determine
            # how many photons there are in each frame.
            n_samples = n_subseries_frames*samples_per_frame
            timestamps = np.zeros(n_samples)
            n_photons_per_frame = np.zeros(n_subseries_frames, dtype=int)
            for j, fn in enumerate(subseries_frame_numbers):
                timestamps[j*samples_per_frame:(j+1)*samples_per_frame] = np.arange(
                    fn*frame_duration_in_sample_unit, (fn+1)*frame_duration_in_sample_unit,
                    sample_step.to(self._sample_unit).value)
                n_photons_per_frame[j] = len(np.where(subseries_photon_frame_numbers == fn)[0])
            n_photons_per_frame = n_photons_per_frame[np.where(n_photons_per_frame > 0)[0]]
            # Get indices of subseries_frame_numbers array corresponding
            # to frames in subseries_photons_frame_numbers array.
            ind = np.arange(n_subseries_frames)[np.in1d(subseries_frame_numbers,
                                                        subseries_photon_frame_numbers)]
            n_photons_gt_0_per_frame = n_photons_per_frame[n_photons_per_frame > 0]
            inds_nested = [[ind[j]]*n_photons_gt_0_per_frame[j]
                           for j in range(len(n_photons_gt_0_per_frame))]
            inds = [item for sublist in inds_nested for item in sublist]
            # Get frames skipped as a function of subseries frame number.
            m = np.insert(subseries_frame_numbers, 0, 0)
            skipped_frames = (m[1:]-m[:-1]-1).cumsum()+1
            # Get number of frames skipped as a function of photon
            # frame number.
            skipped_frames_by_photon = skipped_frames[inds]
            # Get indices of photon times in subseries.
            photon_time_indices = np.rint(
                subseries_incident_photons["time"].data/sample_step.to(
                    subseries_incident_photons["time"].unit).value).astype(
                        int)-skipped_frames_by_photon*samples_per_frame
            # If there are photons assigned to same time index combine
            # them as though they were one photon with an energy equal
            # to the sum of photon energies.
            non_simul_photon_time_indices, non_simul_photon_list_indices, \
              n_photons_per_time_index = np.unique(photon_time_indices, return_index=True,
                                                   return_counts=True)
            if max(n_photons_per_time_index) > 1:
                w = np.where(n_photons_per_time_index > 1)[0]
                subseries_incident_photon_energies = \
                  subseries_incident_photons["energy"][non_simul_photon_list_indices]
                subseries_incident_photon_energies[w] = \
                  [sum(subseries_incident_photons["energy"][non_simul_photon_list_indices[j]:non_simul_photon_list_indices[j]+n_photons_per_time_index[j]])
                   for j in w]
            else:
                subseries_incident_photon_energies = subseries_incident_photons["energy"]
            # Convert photon energies to voltage timeseries
            voltage_deltas = self._convert_photon_energy_to_voltage(
                subseries_incident_photon_energies)
            voltage_delta_timeseries = np.zeros(n_samples)
            voltage_delta_timeseries[non_simul_photon_time_indices] = voltage_deltas.value
            # Convolve voltage delta function time series with voltage
            # pulse shape.
            if self.peak_hold == False:
                # Calculating timestamps of charge resetting and removing 
                # 0th index if it exists
                reset_frames = np.where(subseries_frame_numbers % self.integration_frame == 0)[0]
                reset_frames = reset_frames[subseries_frame_numbers[reset_frames]>0]
                # Accounting for reset_window
                num_reset_steps = np.ceil(self._reset_window/self._sample_step).astype(int)
                # Charge reset acheived by adding in delta function equal and
                # opposite to the sum of the signal since the previous reset
                for fn in reset_frames:
                    voltage_delta_timeseries[fn*samples_per_frame] = np.sum(-voltage_deltas[(non_simul_photon_time_indices<fn*samples_per_frame)\
                                                                                            &(non_simul_photon_time_indices>(fn-self.integration_frame)\
                                                                                              *samples_per_frame)].value)
                    # To account for reset window removing hits that occured during reset
                    for i in range(num_reset_steps):
                        voltage_delta_timeseries[(fn*samples_per_frame)+(i+1)] = 0
                # Producing heaviside step functions
                voltage = np.zeros(n_samples)
                for hit in np.where(voltage_delta_timeseries != 0)[0]:
                    voltage[hit:] += voltage_delta_timeseries[hit]                  
                # Convolving steps with gaussian to create a more realistic
                # signal response
                voltage = np.convolve(voltage, self._voltage_pulse_shape)
                # Trim edges of convolved time series so that peaks of voltage
                # pulses align with photon times.
                start_index = int(np.rint(self._voltage_peaking_time/self._sample_step))
                end_index = int(np.rint(self._voltage_decay_time/self._sample_step))*(-1)+1
                voltage = voltage[start_index:end_index]
            else:
                voltage = np.convolve(voltage_delta_timeseries, self._voltage_pulse_shape)
                # Trim edges of convolved time series so that peaks of voltage
                # pulses align with photon times.
                start_index = int(np.rint(self._voltage_peaking_time/self._sample_step))
                end_index = int(np.rint(self._voltage_decay_time/self._sample_step))*(-1)+1
                voltage = voltage[start_index:end_index]
            # Add leakage current effect if self.peak_hold is False and
            # self.leakage is True
            if (self.peak_hold == False) and (self.leakage == True):
                reset_indexes = reset_frames * samples_per_frame
                voltage += self._leakage_signal_generate(len(voltage), reset_indexes)
            #generate noise in data if self.noise is True
            if self.noise == True:
                noise_voltage = self._convert_photon_energy_to_voltage(self.noise_fwhm)
                voltage += self._noise_generate(len(voltage),-noise_voltage.value)
            # Generate time series from voltage and timestamps.
            self.subseries = pd.DataFrame(
                voltage, index=pd.to_timedelta(timestamps, self._sample_unit),
                columns=["voltage"])
            subseries_voltage_unit = voltage_deltas.unit
            # Calculate how HEXITEC peak-hold would measure photon
            # times and energies.
            subseries_measured_photon_times, subseries_measured_photon_energies, \
            subseries_next_frame_first_energies, subseries_measured_subframe_photon_times = \
              self._convert_voltage_timeseries_to_measured_photons(
                  self.subseries, subseries_voltage_unit, samples_per_frame,
                  threshold=threshold, return_subframe_photon_times=True)
            subseries_measured_photon_times = subseries_measured_photon_times.to(
                incident_photons["time"].unit)
            subseries_measured_photon_energies = subseries_measured_photon_energies.to(
                incident_photons["energy"].unit)
            subseries_next_frame_first_energies = subseries_next_frame_first_energies.to(
                incident_photons["energy"].unit)
            subseries_measured_subframe_photon_times = \
              subseries_measured_subframe_photon_times.to(incident_photons["time"].unit)

            # Add subseries measured photon times and energies to all
            # photon times and energies Quantities excluding any
            # photons from first or last frame.
            w = np.logical_and(
                subseries_measured_photon_times >= subseries_frame_edges[i]*frame_duration,
                subseries_measured_photon_times < subseries_frame_edges[i+1]*frame_duration)
            measured_photon_times = np.append(
                measured_photon_times, subseries_measured_photon_times.value[w])
            measured_photon_energies = np.append(
                measured_photon_energies, subseries_measured_photon_energies.value[w])
            next_frame_first_energies = np.append(
                next_frame_first_energies, subseries_next_frame_first_energies.value[w])
            measured_subframe_photon_times = np.append(
                measured_subframe_photon_times,
                subseries_measured_subframe_photon_times.value[w])
            time2 = timeit.default_timer()
            print("Finished {0}th subseries in {1} s".format(i+1, time2-time1))
            print(" ")
        # Convert results into table and return.
        return Table(
            [Quantity(measured_photon_times, unit=incident_photons["time"].unit),
             Quantity(measured_photon_energies, unit=incident_photons["energy"].unit),
             Quantity(next_frame_first_energies, unit=incident_photons["energy"].unit),
             Quantity(measured_subframe_photon_times, unit=incident_photons["time"].unit)],
            names=("time", "energy", "next frame first energy", "subframe time"))


    def _convert_voltage_timeseries_to_measured_photons(
            self, timeseries, voltage_unit, samples_per_frame,
            threshold=None, return_subframe_photon_times=False):
        """Converts a time series of HEXITEC voltage to measured photons.

        Parameters
        ----------
        timeseries: `pandas.DataFrame`
            Time series of voltage single of single HEXITEC pixel due to photon hits.

        voltage_unit: `astropy.units.core.Unit` or `str`
            Unit of timeseries.  Must be compatible with
            `astropy.units.quantity.Quantity`.

        threshold: `astropy.units.quantity.Quantity`
            Threshold below which photons are not recorded.
            Must be in units of energy or voltage.  If unit is energy,
            threshold refers to photon energy.  If unit is voltage, threshold
            refers to voltage induced in pixel due to a photon hit which is a
            function of the photon's energy.
            Default=None implies a threshold of 0V

        Returns
        -------
        measured_photon_times: `astropy.units.quantity.Quantity`
            HEXITEC measured times of photon hits.

        measured_photon_energies: `astropy.units.quantity.Quantity`
            HEXITEC measured energies of photons.

        next_frame_first_energies: `astropy.units.quantity.Quantity`
            HEXITEC measured energies of first reading in each frame.
            This can represent the S2 signal.

        """
        # Convert timeseries into measured photon list by resampling
        # at frame rate and taking min.
        # In resample command, 'xN' signifies x nanoseconds.
        frame_peaks = timeseries.resample(
            "{0}N".format(round(self.frame_duration.to(u.ns).value))).min().dropna()
        # Get the first value of each frame as the S2 measurement.
        next_frame_first_voltages = timeseries[::samples_per_frame].shift(
            periods=-1, axis=0).fillna(0)

        if return_subframe_photon_times:
            # Get times when peaks occur at subframe time cadence using
            # voltage ramp.
            frame_resampler = timeseries.resample(
                    "{0}N".format(int(self.frame_duration.to(u.ns).value)))#.agg(
                        #{"voltage": np.argmax})
            frame_peak_times = np.zeros(len(frame_resampler))*np.nan
            # LINE CHANGED WHEN CONVERTING INTO PYTHON 3
            frame_peak_times = np.array([np.nan if frame[1].empty
                                         else (frame[1].idxmin())[0].total_seconds()
                                         for frame in frame_resampler])
            # Resample at ramp time cadence.
            ramp_step_secs = self.ramp_step.to("s").value
            rounded_frame_peak_times = np.floor(frame_peak_times/ramp_step_secs)*ramp_step_secs

        # Convert voltages back to photon energies
        if threshold is None:
            threshold = 0.
        else:
            try:
                threshold = self._convert_photon_energy_to_voltage(threshold).value
            except u.UnitConversionError:
                threshold = threshold.to(u.V).value
            except u.UnitConversionError as err:
                err.args = (
                    "'{0}' not convertible to either 'keV' (energy) or 'V' (voltage)".format(
                        threshold.unit),)
                raise err
        w = np.where(frame_peaks["voltage"] < threshold)[0]
        measured_photon_energies = self._convert_voltages_to_photon_energy(
            frame_peaks["voltage"][w].values, voltage_unit)
        next_frame_first_energies = self._convert_voltages_to_photon_energy(
            next_frame_first_voltages["voltage"][w].values, voltage_unit)
        # Determine time unit of pandas timeseries and convert photon
        # times to Quantity.
        frame_duration_secs = self.frame_duration.to("s").value
        rounded_photon_times = np.round(
            frame_peaks.index[w].total_seconds()/frame_duration_secs)*frame_duration_secs
        measured_photon_times = Quantity(rounded_photon_times, unit="s")

        if return_subframe_photon_times:
            # Convert subframe measured photon times for photons above
            # threshold to Quantity.
            measured_subframe_photon_times = Quantity(rounded_frame_peak_times[w], unit="s")
            return measured_photon_times, measured_photon_energies, \
              next_frame_first_energies, measured_subframe_photon_times
        else:
            return measured_photon_times, measured_photon_energies, next_frame_first_energies


    def _define_voltage_pulse_shape(self):
        """
        Defines the normalised shape of voltage pulse with given discrete sampling frequency.
        """
        sample_step = self._sample_step.to(self._sample_unit).value
        # Convert input peaking and decay times to a standard unit.
        voltage_peaking_time = self._voltage_peaking_time.to(self._sample_unit).value
        voltage_decay_time = self._voltage_decay_time.to(self._sample_unit).value
        # Define other Gaussian parameters.
        mu = voltage_peaking_time
        zero_equivalent = 1e-3
        sigma2_peak = -0.5*mu**2/np.log(zero_equivalent)
        sigma2_decay = \
          -0.5*(voltage_peaking_time+voltage_decay_time-mu)**2/np.log(zero_equivalent)
        # Generate time data points for peak and decay phases.
        t_peaking = np.arange(0, voltage_peaking_time, sample_step)
        t_decay = np.arange(voltage_peaking_time,
                            voltage_peaking_time+voltage_decay_time,
                            sample_step)
        voltage_pulse_shape = np.append(np.exp(-(t_peaking-mu)**2./(2*sigma2_peak)),
                                        np.exp(-(t_decay-mu)**2./(2*sigma2_decay)))
        # Need to normalise voltage_pulse_shape if integrating behaviour
        # selected
        if self.peak_hold == False:
            voltage_pulse_shape = voltage_pulse_shape/np.sum(voltage_pulse_shape)
        return voltage_pulse_shape        
    
    def _noise_generate(self, num, noise_fwhm):
        """
        Adds random noise to voltage data
        
        Parameters
        ----------
        num: `int`
            Number of noises values to be generated. 
        noise_std : `astropy.units.quantity.Quantity`
            FWHM of Gaussian distribution used to generate noise values.
            

        Returns
        -------
        noise: `list`
            List of noise values generated from distribution.
        """
        noise_std = noise_fwhm/2.355
        noise = np.random.normal(0, noise_std, num)
        
        return noise
    
    def _leakage_signal_generate(self,size,reset_indexes):
        """
        Generates voltage data that corresponds to capacitor charge leaking
        
        Parameters
        ----------
        size: `int`
            Length of voltage timeseries array for which leakage data needs
            to be generated.
        reset_indexes: `np.array`
            Array of voltage timeseries indexes corresponding to steps in
            which preamplifier charge is reset.
        
        Returns
        -------
        voltage_leakage: `np.array`
            Array of voltage leakage data.
        """
        # Calculating leakage voltage per pixel per timestep
        charge_pixel_step = (self.leakage*(self.pixel_pitch**2)*self._sample_step).to(u.C)
        voltage_pixel_step = (charge_pixel_step / self.capacitor).to(u.V)
        # Adding 0 index to reset_indexes
        reset_indexes = np.append([0],reset_indexes)
        # Generating voltage leakage data
        voltage_leakage = np.zeros(size)
        for i in range(size):
            voltage_leakage[i] = \
                voltage_pixel_step.value*(i-reset_indexes[np.searchsorted(reset_indexes, i, side='right') -1])
        return voltage_leakage
       
    def _detector_parameters_call(self,detector):
        """
        Changes simulation detector parameters to match those of real-life
        detector e.g. HEXITEC
        
        Parameters
        ----------
        detector: `str`
            Input name corresponding to preset detector
        """
        
        detector_dictionary = {
            "hexitec" : {
                "pixel_pitch" : Quantity(250., unit = u.um),
                "noise_fwhm"  : Quantity(0.8, unit = u.keV),
                "peak_hold"   : True,
                "leakage"     : Quantity(300., unit = u.nA/(u.mm**2)),
                "voltage_peaking_time" : Quantity(2., unit = u.us),
                "voltage_decay_time": Quantity(15., unit= u.us),
                "capacitor" : Quantity(15., unit = u.pF) }    
        }
        
        dict_indexes = ["pixel_pitch", "noise_fwhm", "peak_hold", "leakage",
                        "voltage_peaking_time", "voltage_decay_time", 
                        "capacitor"]
        
        dict_variables = [self.pixel_pitch, self.noise_fwhm, self.peak_hold, 
                          self.leakage, self._voltage_peaking_time, 
                          self._voltage_decay_time, self.capacitor]
        
        if detector in detector_dictionary:
            for ind, val in enumerate(dict_indexes):
                if val in detector_dictionary[detector]:
                    dict_variables[ind] = detector_dictionary[detector][val]
        else:
            print('Specified preset detector does not exist. Continuing '\
                  'simulation with default parameters')
    
    def generate_frame_by_frame_image(self):
        """
        Generates frame by frame image of measured photon data

        Returns
        -------
        None.

        """
        frame_duration = self.frame_duration.to(self.measured_photons["time"].unit)
        unique_frame_times = np.unique(self.measured_photons["time"])
        unique_frame_numbers = (unique_frame_times/frame_duration).value.astype(int)
        frame_images = np.zeros((80,80,len(unique_frame_numbers)))
        for ind, val in enumerate(unique_frame_numbers):
            frame_hits = np.where(self.measured_photons["time"]==unique_frame_times[ind])[0]
            for i in frame_hits:
                frame_images[self.measured_photons["x"][i].astype(int),self.measured_photons["y"][i].astype(int),ind] = self.measured_photons['energy'][i]
       
        self.fig, self.ax= plt.subplots(1, 1)
        self.tracker = IndexTracker(self.ax, frame_images,unique_frame_numbers)
        self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
        plt.show()
        
    def generate_voltage_timeseries_plot(self):
        """
        Produces plot of voltage timeseries for last timeseries of last pixel
        in readout ROI.
        """
        
        plt.plot(self.subseries,color='black')
        plt.title("Voltage timeseries: pixel ({0},{1})".\
                  format(self.readout_xpixel_range[1],self.readout_ypixel_range[1]))
        plt.ylabel("Voltage /$\it{V}$")
        plt.xlabel("Time /$\it{%s}$" %self._sample_unit)
        plt.xlim(left=0)
        
    def _convert_photon_energy_to_voltage(self, photon_energy):
        """Determines HEXITEC peak voltage due to photon of given energy.

        Parameters
        ----------
        photon_energy: `astropy.units.quantity.Quantity`
            Photon energy to be converted to HEXITEC voltage.

        Returns
        -------
        voltage: `astropy.units.quantity.Quantity`
            Voltage corresponding to input photon energy.

        """
        charge = (photon_energy.to(u.keV)/self._w_factor.to(u.keV))*Quantity(1.6E-19,unit=u.C)
        voltage = (charge.to(u.C)/self.capacitor.to(u.F)).value*u.V
        return -voltage


    def _convert_voltages_to_photon_energy(self, voltages, voltage_unit):
        """Determines photon energy from HEXITEC peak voltage.

        Parameters
        ----------
        voltage: `astropy.units.quantity.Quantity`
            HEXITEC pixel voltage to be converted to photon energy.

        Returns
        -------
        photon_energy: `astropy.units.quantity.Quantity`
            Photon energy corresponding to input voltage.

        """
        voltage = -Quantity(voltages, unit=voltage_unit)
        charge = (voltage.to(u.V)*self.capacitor.to(u.F)).value*u.C
        photon_energy = (charge/(Quantity(1.6E-19,unit=u.C)))*self._w_factor.to(u.keV)
        return photon_energy

class IndexTracker(object):
    """
    Allows user to scroll through frames of an array image.
    """
    def __init__(self, ax, X,labels):
        """
        Initialises IndexTracker class.

        Parameters
        ----------
        ax : `matplotlib.pyplot.subplot`
            Suplot onto which image is produced.
        X : `3 dimensional np.array'
            Image data to be sliced along third dimension.
        labels : `np.array`
            Frames to which each slice corresponds.

        """
        self.ax = ax
        self.X = X
        self.labels=labels
        rows, cols, self.slices = X.shape
        # Starting display at 0 index
        self.ind = 0
        # Plotting image with max of colorbar given by max of overall array
        self.im = self.ax.imshow(self.X[:, :, self.ind],cmap='binary',vmax=np.max(X))
        plt.colorbar(self.im,label='$\it{keV}$')
        self.update()

    def onscroll(self, event):
        """
        Changes slice index by +/- 1 when event occurs.
        
        Parameters
        ----------
        event : `str`
            String associated with event name; e.g. 'scroll_event'.
        """
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """ 
        Updates matplotlib plot
        """
        self.im.set_data(self.X[:, :, self.ind])
        # Sets plot title; encorporates frame number into title
        self.ax.set_title('80x80 image: Frame %s' %self.labels[self.ind])
        self.im.axes.figure.canvas.draw()


