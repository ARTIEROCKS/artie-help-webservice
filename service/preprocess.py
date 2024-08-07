import logging
from datetime import datetime
import pandas as pd


# Function to load the json data
def sort(json_data):
    is_underscore_id = True
    if len(json_data) > 0:
        if 'student' in json_data[0]:
            if '_id' in json_data[0]['student']:
                is_underscore_id = True
            else:
                is_underscore_id = False

    if is_underscore_id:
        sorted_data = sorted(json_data, key=lambda x: (x['student']['_id'], x['lastLogin'], x['dateTime']))
    else:
        sorted_data = sorted(json_data, key=lambda x: (x['student']['id'], x['lastLogin'], x['dateTime']))

    return sorted_data


# Function to get the first actions of the exercise
def get_first_action(interventions):
    first_actions = {}
    student_id = None
    date_time = None
    last_login = None
    exercise_id = None
    for element in interventions:
        if 'student' in element:
            if '_id' in element['student']:
                student_id = element['student']['_id']
            elif 'id' in element['student']:
                student_id = element['student']['id']
        if 'dateTime' in element:
            try:
                element['dateTime'] = element['dateTime'].replace('T', ' ')
                element['dateTime'] = element['dateTime'][:26]
                date_time = datetime.strptime(element['dateTime'], '%Y-%m-%d %H:%M:%S.%f')
            except Exception as ex:
                logging.error(str(ex))
                print(str(ex))
        if 'lastLogin' in element:
            last_login = element['lastLogin']
        if 'exercise' in element:
            if '_id' in element['exercise']:
                exercise_id = element['exercise']['_id']
            elif 'id' in element['exercise']:
                exercise_id = element['exercise']['id']

        if student_id is not None and date_time is not None and last_login is not None and exercise_id is not None:
            if student_id + '_' + exercise_id + '_' + last_login in first_actions.keys():
                if date_time < first_actions[student_id + '_' + exercise_id + '_' + last_login]:
                    first_actions[student_id + '_' + exercise_id + '_' + last_login] = date_time
            else:
                first_actions[student_id + '_' + exercise_id + '_' + last_login] = date_time

    return first_actions


# Function to write the software interventions in dataframe format
def write_pedagogical_software_interventions_df(interventions, first_actions):
    df_list = []

    for element in interventions:

        student_gender = None
        student_age = None
        total_seconds = None
        student_mother_tongue = 0
        student_competence = 0
        student_motivation = 0

        exercise_skill_paralellism = 0
        exercise_skill_logical_thinking = 0
        exercise_skill_flow_control = 0
        exercise_skill_user_interactivity = 0
        exercise_skill_information_representation = 0
        exercise_skill_abstraction = 0
        exercise_skill_syncronization = 0
        exercise_valid_solution = 0
        exercise_level = 0
        solution_distance_family_distance = 0
        solution_distance_element_distance = 0
        solution_distance_position_distance = 0
        solution_distance_input_distance = 0
        solution_distance_total_distance = 0
        seconds_help_open = 0
        valid_solution = 0
        grade = 0

        exercise_is_evaluation = False
        request_help = False
        finished_exercise = False

        student_id = None
        exercise_id = None
        last_login = None

        if 'student' in element:
            if '_id' in element['student']:
                student_id = element['student']['_id']
            elif 'id' in element['student']:
                student_id = element['student']['id']

        if 'exercise' in element:
            if '_id' in element['exercise']:
                exercise_id = element['exercise']['_id']
            elif 'id' in element['exercise']:
                exercise_id = element['exercise']['id']

        if 'lastLogin' in element:
            last_login = element['lastLogin'].replace('T', ' ')

        # Time calculation between the first action of the exercise and the current action
        if student_id is not None and last_login is not None and exercise_id is not None:
            if student_id + '_' + exercise_id + '_' + last_login in first_actions.keys():
                if 'dateTime' in element:
                    element['dateTime'] = element['dateTime'].replace('T', ' ')
                    element['dateTime'] = element['dateTime'][:26]
                    date_time_obj = datetime.strptime(element['dateTime'], '%Y-%m-%d %H:%M:%S.%f')
                    first_action = first_actions[student_id + '_' + exercise_id + '_' + last_login]
                    difference = (date_time_obj - first_action)
                    total_seconds = difference.total_seconds()

        # Student information
        if 'student' in element:
            if 'gender' in element['student']:
                student_gender = element['student']['gender']
            if 'age' in element['student']:
                student_age = element['student']['age']
            if 'motherTongue' in element['student']:
                student_mother_tongue = element['student']['motherTongue']
            if 'competence' in element['student']:
                student_competence = element['student']['competence']
            if 'motivation' in element['student']:
                student_motivation = element['student']['motivation']

        # Exercise information
        if 'exercise' in element:
            if 'skills' in element['exercise']:
                for skill in element['exercise']['skills']:
                    if skill['name'] == 'Paralelismo':
                        exercise_skill_paralellism = skill['score']
                    elif skill['name'] == 'Pensamiento lógico':
                        exercise_skill_logical_thinking = skill['score']
                    elif skill['name'] == 'Control de flujo':
                        exercise_skill_flow_control = skill['score']
                    elif skill['name'] == 'Interactividad con el usuario':
                        exercise_skill_user_interactivity = skill['score']
                    elif skill['name'] == 'Representación de la información':
                        exercise_skill_information_representation = skill['score']
                    elif skill['name'] == 'Abstracción':
                        exercise_skill_abstraction = skill['score']
                    elif skill['name'] == 'Sincronización':
                        exercise_skill_syncronization = skill['score']
            if 'valid_solution' in element['exercise']:
                exercise_valid_solution = element['exercise']['valid_solution']
            if 'isEvaluation' in element['exercise']:
                exercise_is_evaluation = element['exercise']['isEvaluation']
            if 'level' in element['exercise']:
                exercise_level = element['exercise']['level']

        # Solution distance information
        if 'solutionDistance' in element:
            if 'familyDistance' in element['solutionDistance']:
                solution_distance_family_distance = element['solutionDistance']['familyDistance']
            if 'elementDistance' in element['solutionDistance']:
                solution_distance_element_distance = element['solutionDistance']['elementDistance']
            if 'positionDistance' in element['solutionDistance']:
                solution_distance_position_distance = element['solutionDistance']['positionDistance']
            if 'inputDistance' in element['solutionDistance']:
                solution_distance_input_distance = element['solutionDistance']['inputDistance']
            if 'totalDistance' in element['solutionDistance']:
                solution_distance_total_distance = element['solutionDistance']['totalDistance']

        if 'requestHelp' in element:
            request_help = element['requestHelp']
        if 'secondsHelpOpen' in element:
            seconds_help_open = element['secondsHelpOpen']
        if 'finishedExercise' in element:
            finished_exercise = element['finishedExercise']
        if 'validSolution' in element:
            valid_solution = element['validSolution']
        if 'grade' in element:
            grade = element['grade']

        # Creating  the row of the csv
        df_list.append(
            {'student_gender': student_gender,
             'student_mother_tongue': student_mother_tongue,
             'student_age': student_age,
             'student_competence': student_competence,
             'student_motivation': student_motivation,
             'exercise_skill_paralellism': exercise_skill_paralellism,
             'exercise_skill_logical_thinking': exercise_skill_logical_thinking,
             'exercise_skill_flow_control': exercise_skill_flow_control,
             'exercise_skill_user_interactivity': exercise_skill_user_interactivity,
             'exercise_skill_information_representation': exercise_skill_information_representation,
             'exercise_skill_abstraction': exercise_skill_abstraction,
             'exercise_skill_syncronization': exercise_skill_syncronization,
             'exercise_valid_solution': exercise_valid_solution,
             'exercise_is_evaluation': int(exercise_is_evaluation),
             'exercise_level': exercise_level,
             'solution_distance_family_distance': solution_distance_family_distance,
             'solution_distance_element_distance': solution_distance_element_distance,
             'solution_distance_position_distance': solution_distance_position_distance,
             'solution_distance_input_distance': solution_distance_input_distance,
             'solution_distance_total_distance': solution_distance_total_distance,
             'seconds_help_open': seconds_help_open,
             'finished_exercise': int(finished_exercise),
             'valid_solution': valid_solution,
             'grade': grade,
             'total_seconds': total_seconds,
             'request_help': int(request_help)}
        )

    return pd.DataFrame(df_list)


# Function to transform the received data
def data_transformation(json_data):
    # 1- Sorts the information
    data = sort(json_data)

    # 2- Get the first action of each exercise
    actions = get_first_action(data)

    # 3- Creating the dataframe
    df = write_pedagogical_software_interventions_df(data, actions)

    return df
