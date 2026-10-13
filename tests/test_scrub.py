import importlib.util
import unittest

from understudy.dataset import clean

HAS_SCRUB_DEPS = all(importlib.util.find_spec(m) for m in ("presidio_analyzer", "gliner"))


@unittest.skipUnless(HAS_SCRUB_DEPS, "needs the [dataset] extra")
class Scrub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from understudy.scrub import Scrubber
        cls.scrubber = Scrubber(allow={"bob", "alice"})

    def setUp(self):
        self.scrubber.organizations.clear()

    def clean(self, text, aliases=None):
        return clean(text, [], aliases or {}, self.scrubber)

    def test_scrubs_contact_details(self):
        self.assertEqual(self.clean("mail bob@example.com or join discord.gg/abc123"),
                         "mail [email] or join [link discord.gg]")
        self.assertEqual(self.clean("call me on 070-123 45 67, or mail a@b.se"),
                         "call me on [number], or mail [email]")

    def test_scrubs_street_addresses_before_aliasing(self):
        out = self.clean("Exempelvägen 12B, 123 45 Småby. or storgatan 3", {"exempelvägen": "[place]"})
        self.assertEqual(out, "[address]. or [address]")
        self.assertEqual(self.clean("skicka till Storgatan 1, 111 22 Stockholm tack"),
                         "skicka till [address] tack")

    def test_scrubs_towns(self):
        self.assertEqual(self.clean("jag bor i lerum, ska till kiruna över helgen"),
                         "jag bor i [place], ska till [place] över helgen")

    def test_amounts_are_not_postcodes(self):
        self.assertEqual(self.clean("det kostar 123 45 kr typ"), "det kostar 123 45 kr typ")

    def test_streets_without_a_number_stay(self):
        self.assertEqual(self.clean("vägen var lång, 12 mil på gatan"),
                         "vägen var lång, 12 mil på gatan")

    def test_links_keep_only_their_domain(self):
        self.assertEqual(
            self.clean('here (https://photos.example.com/abc123?x=1). and www.shop.se/item/9! '
                       '<a href="http://a.example.com/x">'),
            'here ([link photos.example.com]). and [link shop.se]! <a href="[link a.example.com]">')

    def test_scrubs_meeting_passwords(self):
        self.assertEqual(self.clean("Password: 1CzQ9y and lösenord=hunter2 for the call"),
                         "Password: [password] and lösenord=[password] for the call")

    def test_names_the_aliases_missed_become_placeholders(self):
        self.assertEqual(self.clean("anna och erik åker till göteborg med robert", {"robert": "bob"}),
                         "[name] och [name] åker till [place] med bob")

    def test_pseudonyms_and_personas_stay(self):
        self.assertEqual(self.clean("Bob och Alice spelar pubg"), "Bob och Alice spelar pubg")

    def test_organisations_are_counted_not_replaced(self):
        self.assertEqual(self.clean("jobbar på volvo cars sen i våras"),
                         "jobbar på volvo cars sen i våras")
        self.assertEqual(self.scrubber.organizations["volvo cars"], 1)


if __name__ == "__main__":
    unittest.main()
