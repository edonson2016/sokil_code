import os

INPUT = "/home/edonson/sokil_code/2025-09-08 11-05-53.lvx"

#59496298 to 59661486
import struct



print(os.path.getsize(INPUT))

with open(INPUT, mode='rb') as f: # b is important -> binary

    #public header block
    sig = f.read(20)
    version_bytes = f.read(4)
    version = int.from_bytes(version_bytes, "little")
    
    print("Signature:", sig)
    print("Version:", hex(version))

    #private header
    frameDuration = f.read(4)
    frameDuration = int.from_bytes(frameDuration, "little")
    print("Frame Duration: ", frameDuration)

    deviceCt = f.read(1)
    print("Device Count: ", deviceCt)

    #Device info
    lidarSign = f.read(16)
    print("Lidar Sign: ", lidarSign)

    HubSign = f.read(16)
    print("Hub Sign: ", HubSign)
    
    deviceIndex = f.read(1)
    print("Device Index: ", deviceIndex)

    deviceType = f.read(1)
    print("Device Type: ", deviceType)

    extrinsicEnable = f.read(1)
    print("Extrinsic Enable: ", extrinsicEnable)

    pitch = f.read(4)
    pitch = struct.unpack("<f", pitch)
    print("Pitch: ", pitch)

    roll = f.read(4)
    roll = struct.unpack("<f", roll)
    print("roll: ", roll)

    yaw = f.read(4)
    yaw = struct.unpack("<f", yaw)
    print("yaw: ", yaw)

    x = f.read(4)
    x = struct.unpack("<f", x)
    print("x: ", x)
    y = f.read(4)
    y = struct.unpack("<f", y)
    print("y: ", y)
    z = f.read(4)
    z = struct.unpack("<f", z)
    print("z: ", z)

    #x = f.read(59661510-88)

    #point cloud (frame header)
    currentOffset = f.read(8)
    currentOffset = int.from_bytes(currentOffset, "little")
    print("Current Offset: ", currentOffset)
    currentOffset = f.read(8)
    currentOffset = int.from_bytes(currentOffset, "little")
    print("Next Offset: ", currentOffset)
    currentOffset = f.read(8)
    currentOffset = int.from_bytes(currentOffset, "little")
    print("Frame Index: ", currentOffset)

    #package header
    deviceIndex = f.read(1)
    print("device index: ", deviceIndex)

    protocolVer = f.read(1)
    print("protocol version: ", protocolVer)

    slotID = f.read(1)
    print("Slot ID, ", slotID)

    lidarID = f.read(1)
    print("Lidar ID, ", lidarID)

    reserved = f.read(1)
    print("reserved: ", reserved)

    statusCode = f.read(4)
    print("Status Code: ", statusCode)

    timeStampType = f.read(1)
    print("time stamp type: ", timeStampType)

    dataType = f.read(1)
    print("data type: ", dataType)

    timeStamp = f.read(8)
    timeStamp = int.from_bytes(timeStamp, "little")
    print("time stamp:", timeStamp)

    firstX = f.read(4)
    firstX = struct.unpack("<f", firstX)
    print("First X: ", firstX)

    firstY = f.read(4)
    firstY = struct.unpack("<f", firstY)
    print("First Y: ", firstY)

    firstZ = f.read(4)
    firstZ = struct.unpack("<f", firstZ)
    print("First Z: ", firstZ)

    reflec = f.read(1)
    print("reflec: ", reflec)
    
    #rawdata = f.read(42*30)
    #print("raw: ", rawdata)
    
    deviceIndex = f.read(1)
    print("device index: ", deviceIndex)

    protocolVer = f.read(1)
    print("protocol version: ", protocolVer)

    slotID = f.read(1)
    print("Slot ID, ", slotID)

    lidarID = f.read(1)
    print("Lidar ID, ", lidarID)

    reserved = f.read(1)
    print("reserved: ", reserved)

    statusCode = f.read(4)
    print("Status Code: ", statusCode)

    timeStampType = f.read(1)
    print("time stamp type: ", timeStampType)

    dataType = f.read(1)
    print("data type: ", dataType)

    timeStamp = f.read(8)
    print("time stamp:", timeStamp)



    