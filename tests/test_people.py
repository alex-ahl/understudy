import unittest

from understudy.people import Persona, read, rename


class People(unittest.TestCase):
    def test_persona_tables_and_flags(self):
        people = read({"persona": {"bob": {"names": ["robert_1987", "Robban"], "profile": ["Likes trains."],
                                           "token": "t"},
                                   "alice": {}},
                       "aliases": {"carl_xx": "carl"}},
                      personas=["dave", "Bob"], aliases=["eve=erin"])
        self.assertEqual(people.personas, [Persona("bob", ["robert_1987", "Robban"], ["Likes trains."], "t"),
                                           Persona("alice"), Persona("dave")])
        self.assertEqual(people.aliases, {"robert_1987": "bob", "robban": "bob", "carl_xx": "carl", "eve": "erin"})

    def test_a_name_renamed_two_ways_is_an_error(self):
        with self.assertRaises(ValueError):
            read({"persona": {"bob": {"names": ["robert"]}}, "aliases": {"Robert": "kim"}})
        read({"aliases": {"robert": "bob", "Robert": "bob"}})

    def test_malformed_alias_flag_is_an_error(self):
        with self.assertRaises(ValueError):
            read({}, aliases=["robert"])

    def test_names_match_whole_words_and_their_possessive(self):
        aliases = {"per": "kim", "robert": "bob"}
        self.assertEqual(rename("per, pers and super; roberts and bigrobert", aliases),
                         "kim, kim and super; bob and bigrobert")


if __name__ == "__main__":
    unittest.main()
