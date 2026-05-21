import struct
import zlib
from io import BytesIO
import numpy as np
from PIL import Image

COMPRESSION_TYPE_COLOR = {
    0: 'raw',
    1: 'jpeg',  # ScanNet typically uses jpeg
    2: 'png',
}

COMPRESSION_TYPE_DEPTH = {
    0: 'raw_ushort',
    1: 'zlib_ushort', # ScanNet typically uses zlib_ushort
    2: 'occlusion_kernel_ushort',
}

# sens_handler.py

import struct
import zlib
from io import BytesIO
import numpy as np
from PIL import Image

class SensorData:
    """
    Parses and holds the data from a .sens file.
    This class reads the entire file upon initialization.
    """
    def __init__(self, filename):
        self.version = 4
        with open(filename, 'rb') as f:
            self._load(f)

    def _load(self, f):
        """ Load data from a binary file stream. """
        version = struct.unpack('I', f.read(4))[0]
        assert version == self.version, f"Unsupported .sens file version: got {version}, expected {self.version}"
        
        strlen = struct.unpack('Q', f.read(8))[0]
        self.sensor_name = f.read(strlen).decode('utf-8')
        
        # Read camera intrinsics and extrinsics
        self.intrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
        self.extrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
        self.intrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
        self.extrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
        
        # Read compression types and image dimensions
        self.color_compression_type = struct.unpack('i', f.read(4))[0] # 1 for jpeg
        self.depth_compression_type = struct.unpack('i', f.read(4))[0] # 1 for zlib_ushort
        self.color_width = struct.unpack('I', f.read(4))[0]
        self.color_height =  struct.unpack('I', f.read(4))[0]
        self.depth_width = struct.unpack('I', f.read(4))[0]
        self.depth_height = struct.unpack('I', f.read(4))[0]
        self.depth_shift = struct.unpack('f', f.read(4))[0]

        declared_frames = struct.unpack('Q', f.read(8))[0]

        self.frames = []
        actual_frames = 0

        for i in range(declared_frames):
            frame_data = {}
            # Check position and read camera_to_world
            data = f.read(16 * 4)
            if len(data) != 64:
                # File truncated - use actual frame count
                print(f"Warning: SENS file truncated at frame {i}/{declared_frames}. Using {actual_frames} frames instead.")
                break

            frame_data['camera_to_world'] = np.asarray(struct.unpack('f' * 16, data)).reshape(4, 4)

            # Try to read timestamps
            timestamp_data = f.read(16)  # 2 * 8 bytes
            if len(timestamp_data) != 16:
                print(f"Warning: SENS file truncated while reading timestamps at frame {i}. Using {actual_frames} frames instead.")
                break
            frame_data['timestamp_color'] = struct.unpack('Q', timestamp_data[:8])[0]
            frame_data['timestamp_depth'] = struct.unpack('Q', timestamp_data[8:])[0]

            # Try to read data sizes
            size_data = f.read(16)  # 2 * 8 bytes
            if len(size_data) != 16:
                print(f"Warning: SENS file truncated while reading data sizes at frame {i}. Using {actual_frames} frames instead.")
                break
            color_data_size = struct.unpack('Q', size_data[:8])[0]
            depth_data_size = struct.unpack('Q', size_data[8:])[0]

            # Try to read actual data
            color_data = f.read(color_data_size)
            if len(color_data) != color_data_size:
                print(f"Warning: SENS file truncated while reading color data at frame {i}. Using {actual_frames} frames instead.")
                break

            depth_data = f.read(depth_data_size)
            if len(depth_data) != depth_data_size:
                print(f"Warning: SENS file truncated while reading depth data at frame {i}. Using {actual_frames} frames instead.")
                break

            # Store the raw compressed data to be decoded on demand
            frame_data['color_data_raw'] = color_data
            frame_data['depth_data_raw_compressed'] = depth_data
            self.frames.append(frame_data)
            actual_frames += 1

        self.num_frames = actual_frames
        if actual_frames != declared_frames:
            print(f"SENS file loading completed: {actual_frames}/{declared_frames} frames successfully loaded.")

    def get_color_image(self, frame_idx):
        """
        Decodes and returns the color image for a specific frame as a PIL Image.
        """
        if not 0 <= frame_idx < len(self.frames):
            raise IndexError("frame_idx is out of bounds.")
        
        # Assuming JPEG compression (type 1)
        raw_data = self.frames[frame_idx]['color_data_raw']
        return Image.open(BytesIO(raw_data))

def save_color_image_from_sens(sens_file_path, frame_idx, output_image_path):
    """
    Function 1: Reads a .sens file, extracts the color image for a given
    frame index, and saves it to the specified path.
    
    Args:
        sens_file_path (str): Path to the .sens file.
        frame_idx (int): The index of the frame to save.
        output_image_path (str): The destination path for the saved image (e.g., 'output.jpg').
    """
    try:
        sensor_data = SensorData(sens_file_path)
        print(f"Successfully loaded '{sens_file_path}' with {len(sensor_data.frames)} frames.")
        
        image = sensor_data.get_color_image(frame_idx)
        image.save(output_image_path)
        print(f"Saved frame {frame_idx} to '{output_image_path}'")
        
    except FileNotFoundError:
        print(f"Error: .sens file not found at '{sens_file_path}'")
    except IndexError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def read_sens_file(filename):
    """
    Reads a .sens file and returns a list of dictionaries, where each dictionary
    contains information for one frame.
    """
    sensor_data = SensorData(filename)
    return sensor_data.frames

if __name__ == '__main__':
    # Please replace 'path_to_your_file.sens' with the path to your .sens file.
    # e.g., './scannet_data/scans/scene0011_00/scene0011_00.sens'
    sens_file_path = './scannet_data/scans/scene0011_00/scene0011_00.sens' 
    
    try:
        # Read the file and get the list of frames
        list_of_frames = read_sens_file(sens_file_path)

        # Print information of the first frame as an example
        if list_of_frames:
            first_frame = list_of_frames[0]
            print(f"Successfully read .sens file, containing {len(list_of_frames)} frames.")
            print("\n--- First Frame Information ---")
            print(f"Frame Index: {first_frame.get('frame_index')}")
            print(f"Color Image Timestamp: {first_frame.get('timestamp_color')}")
            print(f"Depth Image Timestamp: {first_frame.get('timestamp_depth')}")
            print("Camera-to-World Transformation Matrix: \n", first_frame.get('camera_to_world'))
            
            # Show color image information
            color_img = first_frame.get('color_image')
            if color_img:
                print(f"Color Image Dimensions: {color_img.size}")
                # color_img.show() # Uncomment to display the image

            # Show depth image information
            depth_img = first_frame.get('depth_image')
            if depth_img is not None:
                print(f"Depth Image Dimensions: {depth_img.shape}")
                print(f"Depth Image Data Type: {depth_img.dtype}")
                print("Depth Image (top-left 5x5 pixels, unit: meters): \n", depth_img[:5, :5])

        else:
            print("No frames found in the file.")

    except FileNotFoundError:
        print(f"Error: File not found. Please check if the path '{sens_file_path}' is correct.")
    except Exception as e:
        print(f"An error occurred while reading or parsing the file: {e}")