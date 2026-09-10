#!/bin/bash
DATA_DIR=$1

if [ "$#" -eq 0 ]; then
  echo "Error: missing data folder parameter"
  echo "Usage: $0 [data_dir]"
  exit 1
fi

images=$(find "$DATA_DIR" -type f -name \*.dng)

DNG_EXT=".dng"
META_EXT="_meta.json"


for img_path in $images; do
  # extract metadata
    # -j: json format
    # -n: disable human-readable txt conv
    #
  ext/exiftool/exiftool -j -n \
    -IFD0:Make -IFD0:Model -IFD0:UniqueCameraModel \
    -IFD0:ColorMatrix1 -IFD0:ColorMatrix2 \
    -IFD0:CalibrationIlluminant1 -IFD0:CalibrationIlluminant2 \
    -IFD0:ForwardMatrix1 -IFD0:ForwardMatrix2 \
    -IFD0:CameraCalibration1 -IFD0:CameraCalibration2 \
    -IFD0:AnalogBalance -IFD0:AsShotNeutral \
    -SubIFD:ImageWidth -SubIFD:ImageHeight \
    -SubIFD:CFARepeatPatternDim -SubIFD:CFAPattern2 -SubIFD:CFAPlaneColor -SubIFD:CFALayout \
    -SubIFD:BlackLevelRepeatDim -SubIFD:BlackLevel -SubIFD:WhiteLevel \
    -SubIFD:ActiveArea -SubIFD:DefaultCropOrigin -SubIFD:DefaultCropSize \
    "$img_path" > "${img_path/"$DNG_EXT"/"$META_EXT"}"

  # -4: Linear 16-bit color [linear gamma curve and no auto-brightness]
  # -D: Document Mode: colors are not rescaled [totally raw]
  # -T: Output in tiff format
  ext/dcraw -4 -D -T "$img_path"

  echo -ne "Image processed $img_path"\\r

done

echo "Finished Successfully!"