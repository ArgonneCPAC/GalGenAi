import numpy as np
from surveycodex.utilities import mag2counts

def counts_to_magnitude(counts, survey, filter_name):
    """
    Convert flux (in electron counts) to AB magnitude using survey zero points.
    
    This is the analytical inverse of mag2counts from survey_codex.

    
    Parameters:
    -----------
    counts : float or np.ndarray
        Flux in electrons (total counts over exposure)
    survey : surveycodex.Survey
        Survey object (e.g., from get_survey('HSC'))
    filter_name : str
        Filter name (e.g., 'g', 'r', 'i', 'z', 'y')
    
    Returns:
    --------
    magnitude : float or np.ndarray
        AB magnitude
    """
    filter_obj = survey.get_filter(filter_name)
    
    # counts = k * 10^(-mag/2.5), where k = 10^(zeropoint/2.5) * exposure_time
    # This gives us: k = 10^(zeropoint/2.5) * exposure_time
    ref_counts = mag2counts(0.0, survey=survey, filter=filter_obj).value

    mag = -2.5 * np.log10(counts / ref_counts)
    
    return mag