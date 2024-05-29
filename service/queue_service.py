import json
import logging
from repository.db import Database


# Function to transform a txt json to an object
def load_json_data(txt_json_data):
    data = json.loads(txt_json_data)
    return data


# Function to create a new interaction object
def create_new_interaction_object(new_data, is_array=False):
    interactions = []

    # If there are no interactions, we create and insert a new document
    student_id = None
    exercise_id = None
    last_login = None

    element = new_data[0] if is_array else new_data

    if "student" in element:
        if "_id" in element["student"]:
            student_id = element["student"]["_id"]
        elif "id" in element["student"]:
            student_id = element["student"]["id"]

    if "exercise" in element:
        if "_id" in element["exercise"]:
            exercise_id = element["exercise"]["_id"]
        elif "id" in element["exercise"]:
            exercise_id = element["exercise"]["id"]

    if "lastLogin" in element:
        last_login = element["lastLogin"]

    if is_array:
        for item in new_data:
            interactions.append(item)
    else:
        interactions.append(element)

    document = {"student_id": student_id, "exercise_id": exercise_id, "last_login": last_login,
                "interactions": interactions}

    return document


# Function to get all the user interactions from the database
def get_student_interactions(new_data, client=None):

    logging.info("Getting the student interactions")

    # 1- Transforms the txt_json into json
    logging.debug("Transforming the txt into a json")
    new_data = load_json_data(new_data)

    # 2- Extracting the new_data information
    new_data_student_id = None
    new_data_exercise_id = None
    new_data_last_login = None

    # Checks if the data is an array or not
    is_array = isinstance(new_data, list)
    element = new_data[0] if is_array else new_data

    if "student" in element:
        if "_id" in element["student"]:
            new_data_student_id = element["student"]["_id"]
        elif "id" in element["student"]:
            new_data_student_id = element["student"]["id"]
    if "exercise" in element:
        if "_id" in element["exercise"]:
            new_data_exercise_id = element["exercise"]["_id"]
        elif "id" in element["exercise"]:
            new_data_exercise_id = element["exercise"]["id"]
    if "lastLogin" in element:
        new_data_last_login = element["lastLogin"]

    # 2- Searches the information about the student
    logging.debug("Getting information from DB of student id: " + str(new_data_student_id))
    db = Database()
    student_query = {"student_id": new_data_student_id}
    document, client = db.search(student_query, client)

    # 3- Looking if the information is about the same exercise and last login
    if document is not None:

        # 3.1- If the exercise and the last login are the same, we update the object
        if document["exercise_id"] == new_data_exercise_id and document["last_login"] == new_data_last_login:
            interactions = document["interactions"]
            if interactions is None:
                interactions = []

            # If the new_data is an array
            if is_array:
                for obj in new_data:
                    interactions.append(obj)
            else:
                interactions.append(new_data)

            logging.debug("Updating the document ID: " + str(document["_id"]))
            new_values = {"interactions": interactions}
            query = {"_id": document["_id"]}
            result, client = db.update(query, new_values, client)

        # 3.2- If the data are not the same, we first delete the current information and we insert a new document
        else:
            # 3.2.1- Deletes the document
            logging.debug("Deleting the document ID: " + str(document["_id"]))
            query = {"_id": document["_id"]}
            result, client = db.delete(query, client)

            # 3.2.2- Creates a new document to be inserted
            logging.debug("Creating new interaction object")
            document = create_new_interaction_object(new_data, is_array)
            result, client = db.insert(document, client)

    else:
        # 4- If the data does not exist, we insert a new document
        logging.debug("Inserting the document in DB the data of student id: " + str(new_data_student_id))
        document = create_new_interaction_object(new_data, is_array)
        result, client = db.insert(document, client)

    return document, client
