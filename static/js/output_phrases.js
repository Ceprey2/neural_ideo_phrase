          //  var centroids_arr = centroids.split(" ");
            var centroids_subcentroids = new Array();


            console.log("first element from dict_centroid_phrases|safe")
            console.log(var_centroid_phrases[0])

            var parsed_centroid_phrases = JSON.parse(var_centroid_phrases[0]);
              console.log("main_centroid of the first data element")
              console.log(parsed_centroid_phrases["main_centroid"])
            console.log("centroids centroids_arr")

            function parse_k_means_dict(var_centroid_phrases){




                  var_centroid_phrases.forEach(function (element){

                      current_entry = JSON.parse(element);

                    centroids_subcentroids.push(current_entry["main_centroid"])
                  });
            };

              parse_k_means_dict(var_centroid_phrases);

              alert(centroids_subcentroids);



            var_centroid_phrases.forEach(function (entry_element) {
                    console.log("centroids_element");
                    console.log(entry_element);
                    parsed_entry_element = JSON.parse(entry_element)
                    current_centroid = parsed_entry_element["main_centroid"]
                $('#select_centroid').append(`<option value=${current_centroid}>
                                       ${current_centroid}
                                  </option>`)

            });

              console.log("Length var centroid"+var_centroid_phrases.length)
              console.log("Length dict centroid"+dict_centroid_phrases_hierarchical.length)

                  dict_centroid_phrases_hierarchical.forEach(function (entry_element) {
                    console.log("centroids_element_hierarchical");
                    console.log(entry_element);
                    parsed_entry_element = JSON.parse(entry_element)
                    current_centroid = parsed_entry_element["centroid"]
                $('#select_centroid_hierarchical').append(`<option value=${current_centroid}>
                                       ${current_centroid}
                                  </option>`)

                       current_centroid_level2 = parsed_entry_element["subcentroid"]
                       $('#select_centroid_hierarchical_level2').append(`<option value=${current_centroid}>
                                       ${current_centroid}
                                  </option>`)

            });

    function get_phrases_according_descriptors(descriptor1, descriptor2, json_arr){

         var phrases_str_to_return ="";

        json_arr.forEach(function (element) {


            parsed_element = JSON.parse(element);
            // alert('%'+parsed_element["centroid"].trim()+'%'+ " "+'%'+$('#select_centroid').val()+'%');
            // alert('%'+parsed_element["centroid"].trim()+'%');
            if (descriptor1 == parsed_element["centroid"].trim() && (descriptor2 == parsed_element["centroid"].trim() || descriptor2 == "")) {


                // $('#div_ideas').text("Idea:" + parsed_element["centroid"].trim() + "<br/>" + "Related ideas: " + parsed_element["neighbors"]);
                // $('#div_phrases').text($('#div_phrases').text() + " " + parsed_element["ukrlangphrases"]);

                console.log("ukrlangphrases")
                console.log(parsed_element["ukrlangphrases"])
                phrases_str_to_return += parsed_element["ukrlangphrases"]

            }

        });

        return phrases_str_to_return.replace("null", " ");

    }

             $('#select_centroid').change(function () {

                   console.log("$('#select_centroid').val()")
                   console.log($('#select_centroid').val())

                $('#div_ideas').text("ideas");
                $('#div_phrases').text("ukrlangphrases");

                var_centroid_phrases.forEach(function (element) {

                    parsed_element = JSON.parse(element);
                   // alert('%'+parsed_element["centroid"].trim()+'%'+ " "+'%'+$('#select_centroid').val()+'%');
                   // alert('%'+parsed_element["centroid"].trim()+'%');
                   if ($('#select_centroid').val().trim() == parsed_element["centroid"].trim()) {



                       $('#div_ideas').text("Idea:"+parsed_element["centroid"].trim()+"<br/>"+"Related ideas: "+parsed_element["neighbors"]);


                    console.log("ukrlangphrases")
                    console.log(parsed_element["ukrlangphrases"])

                   };




                });
                 $('#div_phrases').text(get_phrases_according_descriptors($('#select_centroid').val().trim(), "", var_centroid_phrases));
             });


    $('#select_centroid_hierarchical_level2').change(function () {

                   console.log("$('#select_centroid_hierarchical_level2').val()")
                   console.log($('#select_centroid_hierarchical_level2').val())



                 $('#div_phrases_hierarchical').text(get_phrases_according_descriptors($('#select_centroid_hierarchical_level2').val().trim(), "", dict_centroid_phrases_hierarchical));
             });

