from patreon.jsonapi.parser import JSONAPIParser, JSONAPIResource, JSONAPIRelationshipInfo, JSONAPIRelatedResource

jsonapi_doc = {
    "links": {
        "self": "http://example.com/articles",
        "next": "http://example.com/articles?page[offset]=2",
        "last": "http://example.com/articles?page[offset]=10"
    },
    "data": [{
        "type": "articles",
        "id": "1",
        "attributes": {
            "title": "JSON API paints my bikeshed!"
        },
        "relationships": {
            "author": {
                "links": {
                    "self": "http://example.com/articles/1/relationships/author",
                    "related": "http://example.com/articles/1/author"
                },
                "data": {"type": "people", "id": "9"}
            },
            "comments": {
                "links": {
                    "self": "http://example.com/articles/1/relationships/comments",
                    "related": "http://example.com/articles/1/comments"
                },
                "data": [
                    {"type": "comments", "id": "5"},
                    {"type": "comments", "id": "12"}
                ]
            }
        },
        "links": {
            "self": "http://example.com/articles/1"
        }
    }],
    "included": [{
        "type": "people",
        "id": "9",
        "attributes": {
            "first-name": "Dan",
            "last-name": "Gebhardt",
            "twitter": "dgeb"
        },
        "links": {
            "self": "http://example.com/people/9"
        }
    }, {
        "type": "comments",
        "id": "5",
        "attributes": {
            "body": "First!"
        },
        "relationships": {
            "author": {
                "data": {"type": "people", "id": "2"}
            }
        },
        "links": {
            "self": "http://example.com/comments/5"
        }
    }, {
        "type": "comments",
        "id": "12",
        "attributes": {
            "body": "I like XML better"
        },
        "relationships": {
            "author": {
                "data": {"type": "people", "id": "9"}
            }
        },
        "links": {
            "self": "http://example.com/comments/12"
        }
    }]
}
parsed = JSONAPIParser(jsonapi_doc)
article_1 = parsed.find_resource_by_type_and_id('articles', '1')


def test_document_has_accessible_raw_data():
    assert parsed.json_data is jsonapi_doc


def test_document_has_accessible_parsed_data():
    parsed_data = parsed.data()
    assert isinstance(parsed_data, list)
    assert len(parsed_data) == 1

    parsed_datum = parsed_data[0]
    assert isinstance(parsed_datum, JSONAPIResource)


def test_document_can_find_by_type_and_id():
    comments_12 = parsed.find_resource_by_type_and_id('comments', '12')
    assert isinstance(comments_12, JSONAPIResource)
    assert comments_12.type() == 'comments'
    assert comments_12.id() == '12'


def test_resource_type():
    assert article_1.type() == 'articles'


def test_resource_id():
    assert article_1.id() == '1'


def test_resource_attributes():
    assert article_1.attributes() == {
        'title': "JSON API paints my bikeshed!"
    }


def test_resource_attribute():
    assert article_1.attribute('title') == "JSON API paints my bikeshed!"


def test_resource_relationships():
    assert article_1.relationships() == {
        "author": {
            "links": {
                "self": "http://example.com/articles/1/relationships/author",
                "related": "http://example.com/articles/1/author"
            },
            "data": {"type": "people", "id": "9"}
        },
        "comments": {
            "links": {
                "self": "http://example.com/articles/1/relationships/comments",
                "related": "http://example.com/articles/1/comments"
            },
            "data": [
                {"type": "comments", "id": "5"},
                {"type": "comments", "id": "12"}
            ]
        }
    }


def test_resource_relationship():
    related_author = article_1.relationship('author')
    assert isinstance(related_author, JSONAPIResource)
    assert related_author.type() == 'people'
    assert related_author.id() == '9'
    assert related_author.attribute('first-name') == 'Dan'
    assert related_author.attribute('last-name') == 'Gebhardt'
    assert related_author.attribute('twitter') == 'dgeb'

    related_comments = article_1.relationship('comments')
    assert isinstance(related_comments, list)

    related_comment_5 = related_comments[0]
    assert isinstance(related_comment_5, JSONAPIResource)
    assert related_comment_5.type() == 'comments'
    assert related_comment_5.id() == '5'
    assert related_comment_5.attribute('body') == 'First!'

    related_comment_12 = related_comments[1]
    assert isinstance(related_comment_12, JSONAPIResource)
    assert related_comment_12.type() == 'comments'
    assert related_comment_12.id() == '12'
    assert related_comment_12.attribute('body') == 'I like XML better'


def test_resource_relationship_info():
    related_author_info = article_1.relationship_info('author')
    assert isinstance(related_author_info, JSONAPIRelationshipInfo)


def test_relationship_info_json_data():
    related_author_info = article_1.relationship_info('author')
    assert related_author_info.json_data == {
        "links": {
            "self": "http://example.com/articles/1/relationships/author",
            "related": "http://example.com/articles/1/author"
        },
        "data": {"type": "people", "id": "9"}
    }


def test_relationship_info_resource():
    related_author_info = article_1.relationship_info('author')
    related_author = related_author_info.resource()
    assert isinstance(related_author, JSONAPIResource)
    assert related_author.attribute('first-name') == 'Dan'
    assert related_author.attribute('last-name') == 'Gebhardt'
    assert related_author.attribute('twitter') == 'dgeb'


def test_relationship_info_resource_reference():
    related_author_info = article_1.relationship_info('author')
    related_author_reference = related_author_info.resource_reference()
    assert isinstance(related_author_reference, JSONAPIRelatedResource)
    assert related_author_reference.type() == 'people'
    assert related_author_reference.id() == '9'
    related_author = related_author_reference.resource()
    assert related_author.attribute('first-name') == 'Dan'
    assert related_author.attribute('last-name') == 'Gebhardt'
    assert related_author.attribute('twitter') == 'dgeb'
