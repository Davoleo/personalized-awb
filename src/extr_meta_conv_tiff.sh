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
MAPS_EXT="_maps.json"


for img_path in $images; do
  # extract metadata
    # -j: json format
    # -n: disable human-readable txt conv
  ext/exiftool/exiftool -j -n \
    -IFD0:Make -IFD0:Model -IFD0:UniqueCameraModel -IFD0:Orientation \
    -IFD0:ColorMatrix1 -IFD0:ColorMatrix2 \
    -IFD0:BaselineExposure -IFD0:LinearResponseLimit \
    -IFD0:ProfileHueSatMapDims -IFD0:ProfileHueSatMapData1 -IFD0:ProfileHueSatMapData2 \
    -IFD0:ProfileLookTableDims -IFD0:ProfileLookTableData \
    -IFD0:ProfileName -IFD0:ProfileEmbedPolicy\
    -IFD0:CalibrationIlluminant1 -IFD0:CalibrationIlluminant2 \
    -IFD0:ForwardMatrix1 -IFD0:ForwardMatrix2 \
    -IFD0:CameraCalibration1 -IFD0:CameraCalibration2 \
    -IFD0:AnalogBalance -IFD0:AsShotNeutral \
    -SubIFD:ImageWidth -SubIFD:ImageHeight -SubIFD:DefaultScale -SubIFD:DefaultCropOrigin -SubIFD:DefaultCropSize \
    -SubIFD:CFARepeatPatternDim -SubIFD:CFAPattern2 -SubIFD:CFAPlaneColor -SubIFD:CFALayout -SubIFD:BayerGreenSplit \
    -SubIFD:BlackLevelRepeatDim -SubIFD:BlackLevel -SubIFD:WhiteLevel \
    -SubIFD:ActiveArea -SubIFD:DefaultCropOrigin -SubIFD:DefaultCropSize \
    -SubIFD:NoiseProfile -IFD0:BaselineNoise \
    -ExifIFD:ExposureTime -ExifIFD:ISO -ExifIFD:FNumber \
    "$img_path" > "${img_path/"$DNG_EXT"/"$META_EXT"}"

  # extract metadata large maps and tables in separate json file
    # -j: json format
    # -m: ignore minor warnings [large array]
    # -b: print binary
  ext/exiftool/exiftool -j -m -b \
  -ProfileHueSatMapData1 -ProfileHueSatMapData2 -ProfileLookTableData \
  -ProfileHueSatMapDims -ProfileLookTableDims \
    "$img_path" > "${img_path/"$DNG_EXT"/"$MAPS_EXT"}"

  # -4: Linear 16-bit color [linear gamma curve and no auto-brightness]
  # -D: Document Mode: colors are not rescaled [totally raw]
  # -T: Output in tiff format
  ext/dcraw -4 -D -T "$img_path"

  echo -ne "Image processed $img_path"\\r

done

echo "Finished Successfully!"