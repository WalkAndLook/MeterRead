# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: detection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x64\x65tection.proto\x12\x06\x64\x65tect\"g\n\x16ObjectDetectionRequest\x12\x15\n\rcurrent_image\x18\x01 \x01(\t\x12\x0e\n\x06thresh\x18\x02 \x01(\x01\x12\x17\n\niou_thresh\x18\x03 \x01(\x01H\x00\x88\x01\x01\x42\r\n\x0b_iou_thresh\"j\n\rDetectionBbox\x12\r\n\x05label\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x01\x12\x0c\n\x04xmin\x18\x03 \x01(\x05\x12\x0c\n\x04ymin\x18\x04 \x01(\x05\x12\x0c\n\x04xmax\x18\x05 \x01(\x05\x12\x0c\n\x04ymax\x18\x06 \x01(\x05\"\x83\x01\n\x17ObjectDetectionResponse\x12\x0e\n\x06result\x18\x01 \x01(\x05\x12*\n\x0bobject_bbox\x18\x02 \x03(\x0b\x32\x15.detect.DetectionBbox\x12\x1a\n\rresult_status\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\x10\n\x0e_result_status\"a\n\x10ReadMeterRequest\x12\x15\n\rcurrent_image\x18\x01 \x01(\t\x12\x0e\n\x06thresh\x18\x02 \x01(\x01\x12\x17\n\niou_thresh\x18\x03 \x01(\x01H\x00\x88\x01\x01\x42\r\n\x0b_iou_thresh\"#\n\x0bpointVector\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\"M\n\x11ReadMeterResponse\x12\x0e\n\x06result\x18\x01 \x01(\x05\x12(\n\x0bvector_list\x18\x02 \x03(\x0b\x32\x13.detect.pointVector2\xa8\x01\n\x0cioper_server\x12T\n\x0fObjectDetection\x12\x1e.detect.ObjectDetectionRequest\x1a\x1f.detect.ObjectDetectionResponse\"\x00\x12\x42\n\tReadMeter\x12\x18.detect.ReadMeterRequest\x1a\x19.detect.ReadMeterResponse\"\x00\x62\x06proto3')



_OBJECTDETECTIONREQUEST = DESCRIPTOR.message_types_by_name['ObjectDetectionRequest']
_DETECTIONBBOX = DESCRIPTOR.message_types_by_name['DetectionBbox']
_OBJECTDETECTIONRESPONSE = DESCRIPTOR.message_types_by_name['ObjectDetectionResponse']
_READMETERREQUEST = DESCRIPTOR.message_types_by_name['ReadMeterRequest']
_POINTVECTOR = DESCRIPTOR.message_types_by_name['pointVector']
_READMETERRESPONSE = DESCRIPTOR.message_types_by_name['ReadMeterResponse']
ObjectDetectionRequest = _reflection.GeneratedProtocolMessageType('ObjectDetectionRequest', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTDETECTIONREQUEST,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.ObjectDetectionRequest)
  })
_sym_db.RegisterMessage(ObjectDetectionRequest)

DetectionBbox = _reflection.GeneratedProtocolMessageType('DetectionBbox', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONBBOX,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.DetectionBbox)
  })
_sym_db.RegisterMessage(DetectionBbox)

ObjectDetectionResponse = _reflection.GeneratedProtocolMessageType('ObjectDetectionResponse', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTDETECTIONRESPONSE,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.ObjectDetectionResponse)
  })
_sym_db.RegisterMessage(ObjectDetectionResponse)

ReadMeterRequest = _reflection.GeneratedProtocolMessageType('ReadMeterRequest', (_message.Message,), {
  'DESCRIPTOR' : _READMETERREQUEST,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.ReadMeterRequest)
  })
_sym_db.RegisterMessage(ReadMeterRequest)

pointVector = _reflection.GeneratedProtocolMessageType('pointVector', (_message.Message,), {
  'DESCRIPTOR' : _POINTVECTOR,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.pointVector)
  })
_sym_db.RegisterMessage(pointVector)

ReadMeterResponse = _reflection.GeneratedProtocolMessageType('ReadMeterResponse', (_message.Message,), {
  'DESCRIPTOR' : _READMETERRESPONSE,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:detect.ReadMeterResponse)
  })
_sym_db.RegisterMessage(ReadMeterResponse)

_IOPER_SERVER = DESCRIPTOR.services_by_name['ioper_server']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OBJECTDETECTIONREQUEST._serialized_start=27
  _OBJECTDETECTIONREQUEST._serialized_end=130
  _DETECTIONBBOX._serialized_start=132
  _DETECTIONBBOX._serialized_end=238
  _OBJECTDETECTIONRESPONSE._serialized_start=241
  _OBJECTDETECTIONRESPONSE._serialized_end=372
  _READMETERREQUEST._serialized_start=374
  _READMETERREQUEST._serialized_end=471
  _POINTVECTOR._serialized_start=473
  _POINTVECTOR._serialized_end=508
  _READMETERRESPONSE._serialized_start=510
  _READMETERRESPONSE._serialized_end=587
  _IOPER_SERVER._serialized_start=590
  _IOPER_SERVER._serialized_end=758
# @@protoc_insertion_point(module_scope)
