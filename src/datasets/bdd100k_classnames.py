weather_classnames = [
    'clear','overcast','snowy','rainy','partly cloudy','foggy'
]

weather_enhance_classnames = [
    'clear','snowy','rainy','partly cloudy', 'overcast',# 'foggy'
]

weather_2_classnames = [
    'clear','snowy','rainy','partly cloudy', 'overcast', 'foggy'
]

scene_classnames = [
    'highway', 'residential area', 'city street', 'parking lot', 'tunnel'
]

weather_control_classnames = [
    'clear','overcast','snowy','rainy','partly cloudy', 'NA'
]

weather_enhance_control_classnames = [
    'clear','snowy','rainy','partly cloudy', 'NA', #'foggy'
]

weather_2_control_classnames = [
    'clear','snowy','rainy','partly cloudy', 'overcast', 'NA'
]

scene_control_classnames = [
    'highway', 'residential area', 'city street', 'parking lot', 'NA'
]

def get_classnames(source):
    if source == 'weather':
        return weather_classnames
    elif source == 'weather_enhance':
        return weather_enhance_classnames
    elif source == 'weather_2':
        return weather_2_classnames
    elif source == 'scene':
        return scene_classnames
    elif source == 'weather_c':
        return weather_control_classnames
    elif source == 'weather_enhance_c':
        return weather_enhance_control_classnames
    elif source == 'weather_2_c':
        return weather_2_control_classnames
    elif source == 'scene_c':
        return scene_control_classnames
    else:
        raise ValueError(f'Unknown classname source for Bdd100k: {source}')
    return all_classnames