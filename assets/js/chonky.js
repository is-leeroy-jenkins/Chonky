/******************************************************************************************
    Assembly:                Chonky
    Filename:                chonky.js
    Author:                  Terry D. Eppler
    Created:                 06-14-2026

    Last Modified By:        Terry D. Eppler
    Last Modified On:        06-14-2026
    ******************************************************************************************
<copyright file="chonky.js" company="Terry D. Eppler">

    chonky.js
    Copyright © 2026 Terry Eppler

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the “Software”),
    to deal in the Software without restriction,
    including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    You can contact me at: terryeppler@gmail.com or eppler.terry@epa.gov

</copyright>
<summary>
    Provides lightweight browser-side behavior for the Chonky documentation site.

    Purpose:
        Adds conservative documentation-page enhancements for Material for MkDocs,
        including external-link handling that opens off-site references in a new tab
        with safe `noopener noreferrer` behavior. The script is intentionally small so
        it does not interfere with MkDocs navigation, search, or generated API pages.
</summary>
 ******************************************************************************************/
( function()
{
	"use strict";
	
	function markExternalLinks()
	{
		const links = document.querySelectorAll( "a[href^='http']" );
		links.forEach( function( link )
		{
			const href = link.getAttribute( "href" ) || "";
			const isExternal = !href.includes( window.location.hostname );
			if( isExternal )
			{
				link.setAttribute( "target", "_blank" );
				link.setAttribute( "rel", "noopener noreferrer" );
			}
		} );
	}
	
	function initializeChonkyDocs()
	{
		markExternalLinks();
	}
	
	if( document.readyState === "loading" )
	{
		document.addEventListener( "DOMContentLoaded", initializeChonkyDocs );
	}
	else
	{
		initializeChonkyDocs();
	}
	document.addEventListener( "DOMContentSwitch", initializeChonkyDocs );
} )();