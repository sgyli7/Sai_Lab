"""Keyboard curricula built from the actual paired player input programs."""
import copy
from sim2sim.standalone.sprint import templates


def keyboard_courses(include_ordinary=False):
    courses=list(templates().items())
    if include_ordinary:
        paired=[]
        for name,original in courses:
            case=copy.deepcopy(original)
            for segment in case['segments']:
                segment['held']=[key for key in segment['held'] if key!='sprint']
            paired.append((name+'_ordinary',case))
        courses.extend(paired)
    return courses
