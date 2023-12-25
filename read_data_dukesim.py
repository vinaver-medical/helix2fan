import numpy as np
import tqdm
import os
import struct


def unpack_tag(data, tag):
    return struct.unpack('f', data[tag].value)[0]

def read_dicom(parser):
    """ Read DICOM-CT-PD projection data and header information following the DICOM-CT-PD User Manual Version 3
    of the TCIA LDCT-and-Projection-data.

    :param parser: parser containing projection information
    :return: the raw projection data, a parser that contains all relevant DICOM header
    parameters for rebinning and reconstruction
    """
    args = parser.parse_args()
    indices = slice(args.idx_proj_start, args.idx_proj_stop)

    param_dict = {}
    with open('rtcat_params.txt') as param_file:
        for line in param_file:
            line = line.strip()
            key = line.split(':')[0].strip()
            value = line.split(':')[1].strip()
            param_dict[str(key)] = str(value)

    with open('result000_XCAT_EID_xcat_abdomen_thorax_small.xcat') as file:
        projections = np.fromfile(file, dtype=np.float32)

    raw_projections = projections.reshape((-1, 64, 900))
    # raw_projections = raw_projections[:int(len(raw_projections)/4)]
    # raw_projections = np.swapaxes(raw_projections, 2, 1)

    # Read geometry information from the DICOM headers following instructions from the
    # TCIA (LDCT-and-Projection-data) DICOM-CT-PD User Manual Version 3.
    angles = np.linspace(0, 2*np.pi*4, num=1152*4, endpoint=False) #np.array([unpack_tag(d, 0x70311001) for d in data_headers]) + (np.pi / 2)
    angles = - np.unwrap(angles) - np.pi  # Different definition of angles (monotonously increasing, starting from a negative value)
    # dangles = np.array([unpack_tag(d, 0x7033100B) for d in data_headers])  # Flying focal spot dphi
    # dz = np.array([unpack_tag(d, 0x7033100C) for d in data_headers])  # Flying focal spot dz
    # drho = np.array([unpack_tag(d, 0x7033100D) for d in data_headers])  # Flying focal spot drho
    nu = 900  #data_headers[0].Rows
    nv = 64  #data_headers[0].Columns
    du = 1  #unpack_tag(data_headers[0], 0x70291002)  # DetectorElementTransverseSpacing
    dv = 1  #unpack_tag(data_headers[0], 0x70291006)  # DetectorElementAxialSpacing
    dv_rebinned = 1  # [mm] Detector pixel v width of rebinned sinogram.
    det_central_element = np.array((450, 32))  #np.array(struct.unpack('2f', data_headers[0][0x70311033].value))
    dso = 575  #unpack_tag(data_headers[0], 0x70311003)  # DetectorFocalCenterRadialDistance
    dsd = 1050  #unpack_tag(data_headers[0], 0x70311031)  # ConstantRadialDistance
    ddo = dsd - dso#(unpack_tag(data_headers[0], 0x70311031) - unpack_tag(data_headers[0], 0x70311003))  # ConstantRadialDistance - DetectorFocalCenterRadialDistance
    pitch = 1.2  #((unpack_tag(data_headers[-1], 0x70311002) -
             # unpack_tag(data_headers[0], 0x70311002)) /
             #((np.max(angles) - np.min(angles)) / (2 * np.pi)))  # Mayo does not include the tag TableFeedPerRotation, we manually compute the pitch
    z_positions = np.linspace(163.24, 0, 1152*4) #np.array([unpack_tag(d, 0x70311002) for d in data_headers])  # DetectorFocalCenterAxialPosition
    nz_rebinned = abs(int((z_positions[-1] - z_positions[0]) / dv_rebinned))
    hu_factor = 0.1962  #float(data_headers[0][0x70411001].value)  # WaterAttenuationCoefficient (see manual for HU conversion)
    rotview = int(len(angles) / ((angles[-1] - angles[0]) / (2 * np.pi)))

    # Create parser.
    parser.add_argument('--indices', type=int, default=[indices.start, indices.stop],
                        help='Index range of loaded and processed helical projections.')
    parser.add_argument('--nu', type=int, default=nu,
                        help='Number of scanner detector rows [].')
    parser.add_argument('--nv', type=int, default=nv,
                        help='Number of scanner detector columns [].')
    parser.add_argument('--du', type=float, default=du,
                        help='Scanner detector pixel spacing in row direction [mm].')
    parser.add_argument('--dv', type=float, default=dv,
                        help='Scanner detector pixel spacing in column direction [mm].')
    parser.add_argument('--dv_rebinned', type=float, default=dv_rebinned,
                        help='Detector pixel spacing in column direction on virtual rebinned detector [mm].')
    parser.add_argument('--det_central_element', type=float, default=det_central_element.tolist(),
                        help='Central element of the detector. Float index between real detector pixels [].')
    parser.add_argument('--dso', type=float, default=dso,
                        help='Source-object (isocenter) distance [mm].')
    parser.add_argument('--dsd', type=float, default=dsd,
                        help='Source-detector distance [mm].')
    parser.add_argument('--ddo', type=float, default=ddo,
                        help='Detector-object distance [mm].')
    parser.add_argument('--pitch', type=float, default=pitch,
                        help='Pitch of helical trajectory [mm].')
    parser.add_argument('--nz_rebinned', type=int, default=nz_rebinned,
                        help='Number of detector rows on the rebinned detector.')
    parser.add_argument('--rotview', type=int, default=rotview,
                        help='Number of projections of rebinned detector, i.e., number of helical projections within [0, 2pi].')
    parser.add_argument('--hu_factor', type=float, default=hu_factor,
                        help='Water attenuation coefficient to convert the absorption coefficients to HU values [].')
    parser.add_argument('--angles', type=float, default=angles.tolist(),
                        help='Angles of helix projections [rad].')
    parser.add_argument('--z_positions', type=float, default=z_positions.tolist(),
                        help='Axial positions of projections of the helical trajectory [mm].')
    # parser.add_argument('--dangles', type=float, default=dangles.tolist(),
    #                     help='Flying focal spot correction dphi [rad].')
    # parser.add_argument('--dz', type=float, default=dz.tolist(),
    #                     help='Flying focal spot correction dz [mm].')
    # parser.add_argument('--drho', type=float, default=drho.tolist(),
    #                     help='Flying focal spot correction drho [mm].')

    return raw_projections, parser