# -*- coding: utf-8 -*-
"""
Omschrijving: Dit script haalt een access token op van de HL API en beheert deze,
met behulp van client credentials (client ID en secret). Het access token is vereist
voor het authenticeren van aanvragen aan de HL API.
 
Vereisten: een geldig client ID en client secret moeten worden opgegeven.
 
HydroLogic, Amersfoort
Mei 2026
"""

import datetime
import requests
import base64
import jwt


# ============================================================
# HL API AUTHENTICATION
# ============================================================


class HLAuthClient:

    def __init__(self, client_id: str, client_secret: str):
        
        # initialiseert de HLAuthClient met de benodigde authenticatiegegevens
        #
        # Args:
        #    client_id : str 
        #       de client ID voor authenticatie
        #    client_secret : str
        #       de client secret voor authenticatie
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = "https://login.hydronet.com/auth/realms/hydronet/protocol/openid-connect/token"
        self.token = None

    def execute(self):

        # genereert een HTTP-header met een geldig access token voor API-aanvragen
        #   - deze methode controleert eerst of het huidige token nog geldig is.
        #   - indien het token is verlopen (of niet aanwezig is), wordt automatisch een nieuw token opgehaald.
        # 
        # Returns:
        #    Optional[Dict[str, str]]: 
        #        een HTTP-header dictionary met 'Content-Type' en 'Authorization' velden
        # 
        # Raises:
        #    ValueError: Als er geen geldig token kan worden opgehaald.
        
        # stap 1: get a valid token, refreshing if necessary
        if self.get_token_expired():
            token = self.get_token()

            if not token:
                raise ValueError("Failed to retrieve a valid token.")
                
            # step 2: create an HTTP header with the valid access token for API requests
            return {"content-type": "application/json", "Authorization":
                    "Bearer " + token}
        return None
    
    def get_token_expired(self) -> bool:
        
        # controleert of het huidige access token is verlopen.
        #
        # Returns:
        #     bool: True als het token is verlopen of niet aanwezig is, False anders
        
        if not self.token:
            return True

        token_decoded = jwt.decode(self.token, options={"verify_signature": False})
        token_exp_datetime = datetime.datetime.fromtimestamp(token_decoded["exp"], tz=datetime.UTC)
        current_datetime = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(minutes=1)
        return current_datetime > token_exp_datetime
     
    def get_token(self):

        # Haalt een nieuw access token op via de HL API met behulp van de client credentials
        #
        # Returns:
        #    Optional[str]: Het access token indien succesvol opgehaald, anders None
        
        # encode client ID and secret for authentication
        authorization = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("ISO-8859-1")).decode("ascii")
            
        # create authorization header
        headers = { "Authorization": f"Basic {authorization}", "Content-Type": "application/x-www-form-urlencoded"}
        # create authorization body
        body = {"grant_type": "client_credentials"}

        # make the POST request
        response = requests.post(self.auth_url, data=body, headers=headers)
        response.raise_for_status()  # raise an exception for HTTP errors
        
        # parse and return the access token
        self.token = response.json().get("access_token")
        return self.token