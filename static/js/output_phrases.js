          //  var centroids_arr = centroids.split(" ");



            console.log("first element from dict_centroid_phrases|safe")
            console.log(var_centroid_phrases[0])

            var parsed_centroid_phrases = JSON.parse(var_centroid_phrases[0]);
              console.log("main_centroid of the first data element")
              console.log(parsed_centroid_phrases["main_centroid"])
            console.log("centroids centroids_arr")

          function get_main_centroids_from_dict (dict){

                  var descriptors_set = new Set();
                  var subdescriprotrs_set = new Set();


                  dict.forEach(function (element){

                      //alert(element);

                      current_entry = JSON.parse(element);

                      current_entry['entries'].forEach(function (subelement){

                          current_subentry = JSON.parse(subelement);

                           subdescriprotrs_set.add(current_subentry["centroid"]);
                          // alert("subcentroid " + current_subentry["centroid"])
                      });

                    descriptors_set.add(current_entry["main_centroid"])
                  });

                  return [descriptors_set, subdescriprotrs_set];
            };



              var centroids_subcentroids_k_means = get_main_centroids_from_dict(var_centroid_phrases);

              alert(centroids_subcentroids_k_means);



     var centroids_subcentroids_hierarchical = get_main_centroids_from_dict(dict_centroid_phrases_hierarchical);

              alert(centroids_subcentroids_hierarchical);

              function append_descriptors_to_select (select_id, descriprors_array){


                  descriprors_array.forEach(function (current_descriptor) {




                $('#'+select_id).append(`<option value=${current_descriptor}>
                                       ${current_descriptor}
                                  </option>`)

            });



              }
            append_descriptors_to_select ("select_centroid", centroids_subcentroids_k_means[0]);
            append_descriptors_to_select ("select_subcentroid", centroids_subcentroids_k_means[1]);

              append_descriptors_to_select ("select_centroid_hierarchical", centroids_subcentroids_hierarchical[0]);
              append_descriptors_to_select ("select_centroid_hierarchical_level2", centroids_subcentroids_hierarchical[1]);

              console.log("Length var centroid"+var_centroid_phrases.length)
              console.log("Length dict centroid hierarchical"+dict_centroid_phrases_hierarchical.length)

    function get_phrases_according_descriptors(descriptor1, descriptor2, json_arr){

         var phrases_str_to_return ="";

        json_arr.forEach(function (element) {
            parsed_element = JSON.parse(element);
            // alert('%'+parsed_element["centroid"].trim()+'%'+ " "+'%'+$('#select_centroid').val()+'%');
            // alert('%'+parsed_element["centroid"].trim()+'%');
            if (descriptor1 == parsed_element["centroid"].trim() && (descriptor2 == parsed_element["centroid"].trim() || descriptor2 == "")) {

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

