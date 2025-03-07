import patreon
import os

# needed by sphinx
def setup(app):
    return

def scrape_patreon():
    creator_id = os.environ.get('CREATOR_ID') # Creator's Access Token from https://www.patreon.com/portal/registration/register-clients
    if creator_id:
        api_client = patreon.API(creator_id)
        #data = api_client.fetch_campaign_and_patrons().json_data
        #patron_count = data['data'][0]['attributes']['patron_count']
        #print("patron count:", patron_count)

        # Get list of all patrons
        campaign_id = api_client.fetch_campaign().data()[0].id()
        pledges_response = api_client.fetch_page_of_pledges(campaign_id, 50) # 2nd arg is number of pledges per page
        names = []
        for pledge in pledges_response.data():
            patron_id = pledge.relationship('patron').id()
            patron = pledges_response.find_resource_by_type_and_id('user', patron_id)
            full_name = patron.attribute('full_name')
            # Manual substitutions to make it look nicer
            full_name = full_name.replace("Jon Kraft, Analog Devices", "Jon Kraft")
            full_name = full_name.replace("vince baker", "Vince Baker")
            full_name = full_name.replace("meh", "MH")
            if full_name == "Дмитрий Ступаков":
                continue
            if full_name == "Al Grant":
                names.append('Al Grant <img width="15px" height="12px" src="https://pysdr.org/_static/kiwi-bird.svg">')
                continue
            if full_name == "Hash":
                names.append('<a href="https://www.youtube.com/@RECESSIM" style="border-bottom: 0;" target="_blank">Hash <img width="15px" height="12px" src="https://pysdr.org/_static/hash.svg"></a>')
                continue
            names.append(full_name) # there's also 'first_name' which might be better for a public display name
        # Patreon Supporters
        html_string = ''
        html_string += '<div style="font-size: 120%; margin-top: 5px; color: #444;">A big thanks to all PySDR<br><a href="https://www.patreon.com/PySDR" target="_blank">Patreon</a> supporters:</div>'
        html_string += '<div style="font-size: 120%; margin-bottom: 80px; margin-top: 0px; color: #444;">'
        for name in names:
            html_string += '&#9900; ' + name + "<br />"
        # Organizations that are sponsoring (Manually added to get logo included)
        html_string += '<div style="margin-top: 5px; color: #444;">and organization-level supporters:</div>'
        html_string += '<img width="12px" height="12px" src="https://pysdr.org/_static/adi.svg">' + ' <a style="border-bottom: 0;" target="_blank" href="https://www.analog.com/en/design-center/reference-designs/circuits-from-the-lab/cn0566.html">Analog Devices, Inc.</a>' + "<br />"
        html_string += "</div>"
        with open("_templates/patrons.html", "w") as patron_file:
            patron_file.write(html_string)
    else:
        print("\n=====================================================")
        print("Warning- CREATOR_ID wasn't set, skipping patron list")
        print("=====================================================\n")
        with open("_templates/patrons.html", "w") as patron_file:
            patron_file.write('')
