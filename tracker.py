import math

class Trackers:
    def __init__(self):
        # Stocker les positions centrales des objets
        self.center_points = {}

        # chaque fois qu'un nouvel objet est détecté, le count est incrementé
        self.id_count = 0


    def update(self, objects_rect):

        objects_bbs_ids = []

        # Obtenir le point central d'un nouvel objet
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Vérifier si cet objet a déjà été détecté
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break

            # Un nouvel objet est détecté, nous attribuons l'ID à cet objet.
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        # Nettoyer le dictionnaire par points centraux pour supprimer les IDS qui ne sont plus utilisés
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Mise à jour du dictionnaire avec les IDs non utilisés supprimés
        self.center_points = new_center_points.copy()
        return objects_bbs_ids