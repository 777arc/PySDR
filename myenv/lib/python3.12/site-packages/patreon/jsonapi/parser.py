import itertools


class JSONAPIParser(object):
    """This is a reader for JSON API Documents"""
    def __init__(self, json_data):
        self.json_data = json_data

    def data(self):
        if isinstance(self.json_data["data"], list):
            return [JSONAPIResource(datum, self) for datum in self.json_data["data"]]
        else:
            return JSONAPIResource(self.json_data["data"], self)

    def find_resource_by_type_and_id(self, resource_type, resource_id):
        for resource_json_data in self._all_resource_json_data():
            if resource_json_data.get("type") == resource_type and resource_json_data.get("id") == resource_id:
                return JSONAPIResource(resource_json_data, self)
        return None

    def _all_resource_json_data(self):
        data = self.json_data["data"]
        if not isinstance(data, list):
            data = [data]
        resources = itertools.chain([], data)

        if "included" in self.json_data:
            resources = itertools.chain(resources, self.json_data["included"])

        return resources


class JSONAPIResource(object):
    """Represents a single resource in a JSON API Document"""
    def __init__(self, json_data, document):
        self.json_data = json_data
        self.document = document

    def id(self):
        return self.json_data.get("id")

    def type(self):
        return self.json_data.get("type")

    def attributes(self):
        return self.json_data.get("attributes", {})

    def attribute(self, name):
        return self.attributes().get(name, None)

    def relationships(self):
        return self.json_data.get("relationships", {})

    def relationship(self, relationship_name):
        relationship_info = self.relationship_info(relationship_name)
        return relationship_info.resource() if relationship_info else None

    def relationship_info(self, relationship_name):
        relationships = self.relationships()
        if relationship_name not in relationships:
            return None

        return JSONAPIRelationshipInfo(relationships[relationship_name], self.document)


class JSONAPIRelationshipInfo(object):
    """Represents a named relationship in a JSON API Document"""
    def __init__(self, json_data, document):
        self.json_data = json_data
        self.document = document

    def resource(self):
        resource_reference = self.resource_reference()

        if isinstance(resource_reference, list):
            return [one_id.resource() for one_id in resource_reference]

        return resource_reference.resource()

    def resource_reference(self):
        if "data" in self.json_data:
            if isinstance(self.json_data["data"], list):
                return [JSONAPIRelatedResource(datum, self.document) for datum in self.json_data["data"]]
            else:
                return JSONAPIRelatedResource(self.json_data["data"], self.document)


class JSONAPIRelatedResource(object):
    """Represents a single resource related to another resource in a JSON API Document"""
    def __init__(self, json_data, document):
        self.json_data = json_data
        self.document = document

    def type(self):
        return self.json_data.get("type")

    def id(self):
        return self.json_data.get("id")

    def resource(self):
        return self.document.find_resource_by_type_and_id(self.type(), self.id())
