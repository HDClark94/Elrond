# make sure python-neo is installed in the virtual environment buy navigating within /python-neo and installing with "pip install ."
# Note: this python-neo is a fork of the original that affords can read OpenEphys legacy data without .event films
import os
from neuroconv.utils.dict import load_dict_from_file
from neuroconv.datainterfaces import OpenEphysRecordingInterface
from neuroconv import ConverterPipe
import pandas as pd
import numpy as np
from dateutil import tz
import uuid
from datetime import datetime
from pynwb import NWBHDF5IO, NWBFile, TimeSeries


def get_age_at_recording(session_start_time, date_of_birth):
    age = session_start_time - date_of_birth
    age_in_days = age.total_seconds()/(60*60*24)
    return int(age_in_days)


def get_dob(subject_id, dob_csv):
    dob = dob_csv[dob_csv["id"] == subject_id]["dob"].iloc[0]
    dob.to_pydatetime()
    return dob


def convert(recording_path, processed_folder_name, **kwargs):
    # handle save locations and paths
    recording_name = os.path.basename(recording_path)
    nwb_path = recording_path+"/"+processed_folder_name+"/nwb/" # Where to save the .nwb file
    nwb_file_name = nwb_path+recording_name+".nwb"
    if not os.path.exists(nwb_path):
        os.mkdir(nwb_path)

    # handle metadata
    metadata_path = "/mnt/datastore/Harry/basic_metadata.yml" # basic
    # TODO put Additional_files/basic_metadata.yml into a location that can be read from and read from there.
    path_to_dob_csv = "/mnt/datastore/Harry/Grid_anchoring_eLife_2023/mouse_dobs.csv"
    # TODO date of birth information needs to be supplied somewhere.

    params = load_dict_from_file(recording_path+"/params.yml")
    subject_id = recording_path.split("/")[-1].split("_")[0]
    session_id = recording_path.split("/")[-1]
    session_description = params["recording_type"]
    st = ("-".join(session_id.split("_")[2:])).split("-")
    st = datetime(int(st[0]),int(st[1]), int(st[2]), int(st[3]),int(st[4]), int(st[5]), tzinfo=tz.gettz("UK/GMT"))
    dob_csv = pd.read_csv(path_to_dob_csv, parse_dates=["dob"])

    if subject_id in np.asarray(dob_csv["id"]):
        if not os.path.exists(nwb_file_name):
            date_of_birth = get_dob(subject_id, dob_csv)
            age = get_age_at_recording(st, date_of_birth)
            metadata = load_dict_from_file(file_path=metadata_path)
            metadata["NWBFile"]["identifier"] = str(uuid.uuid4())
            metadata["NWBFile"]["session_id"] = session_id
            metadata["NWBFile"]["session_start_time"] = st
            metadata["NWBFile"]["session_description"] = session_description
            metadata["Subject"]["subject_id"] = subject_id
            metadata["Subject"]["age"] = "P"+str(age)
            metadata["Subject"]["date_of_birth"] = date_of_birth
            metadata["Ecephys"]["ElectricalSeries"]["starting_time"] = 0

            # load ephys and convert to nwb
            ephys_interface = OpenEphysRecordingInterface(folder_path=recording_path, stream_name="Signals CH", verbose=True)
            converter = ConverterPipe(data_interfaces=[ephys_interface], verbose=True) # add interfaces to the list if needed
            converter.run_conversion(nwbfile_path=nwb_file_name, metadata=metadata, overwrite=True)

            # append nwb with behaviour data
            io = NWBHDF5IO(nwb_file_name, mode="a")
            nwbfile = io.read()

            behaviour_csv = pd.read_csv(recording_path + "/processed/position_data.csv")
            timestamps = np.asarray(behaviour_csv["time_seconds"])
            del behaviour_csv["time_seconds"]
            for column in list(behaviour_csv):
                if "Unnamed" in column:
                    continue # do nothing with this junk column
                else:
                    behavioral_time_series = TimeSeries(name=column,
                                                        data=np.asarray(behaviour_csv[column]),
                                                        timestamps=timestamps,
                                                        description="position: cm, speed: cm/s, angle: degrees, trials: non-zero indexed, "
                                                                    "trial types: 0-beaconed, 1-nonbeaconed, 2-probe",
                                                        unit="n/a")
                    nwbfile.add_acquisition(behavioral_time_series)
            io.write(nwbfile)
            io.close()
            print("I have completed nwb conversion for ", recording_name)
    return


def main():
    print("")

if __name__ == '__main__':
    main()
